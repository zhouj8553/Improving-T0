import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import logging
import numpy
from tqdm import tqdm
import numpy as np
from collections import Counter
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Type, Union
from datasets import load_dataset
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            logger.info("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings
    
    def similarity(self, queries: Union[str, List[str]], 
                    keys: Union[str, List[str], ndarray], 
                    device: str = None) -> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities
    def process_emb(self, embs, batch_size=64,hidden_size=768):
        # 用batch的形式装data, 并且要归一化
        embs = embs.astype(np.float32)
        embs = embs / np.linalg.norm(embs,axis=1,keepdims=True)
        # embs = embs.to(device)
        # with torch.no_grad():
        #     total_batch = len(embs) // batch_size + (1 if len(embs) % batch_size > 0 else 0)
        #     embs_ = np.zeros((total_batch,batch_size,hidden_size))
        #     for batch_id in range(total_batch):
        #         if (batch_id == total_batch - 1) and (len(embs) % batch_size > 0):
        #             embs_[-1,:,:] = embs[-(len(embs) % batch_size):]
        #         else:
        #             embs_[batch_id,:,:] = embs[batch_id*batch_size:(batch_id+1)*batch_size]
        return embs

    def build_index_emb(self, file_path: Union[str, dict],
                        emb_path: Union[str, dict], 
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64,
                        n_gpus: int = 8,
                        faiss_compress: bool = False):
        DEBUG = True
        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True 
            except:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        # 和底下的函数不一样，如果是 dict 则代表是{file_path:数据/embs}
        if isinstance(file_path, dict):
            logger.info("Building index...")
            embeddings = np.concatenate([emb for path, emb in emb_path.items()])
            sentences_or_file_path = [sent for path, sents in file_path.items() for sent in sents]
            logger.info(f"Search length:{len(embeddings)}")
            self.index = {"sentences": sentences_or_file_path}
        else:
            raw_ds = load_dataset("json", data_files=file_path, cache_dir="/share/zongyu/cache/huggingface/datasets")
            raw_ds = raw_ds['train']
            sentences_or_file_path = [d['inputs_pretokenized'] + d['targets_pretokenized'] for d in raw_ds]
            embeddings = torch.load(emb_path)
            # TODO 用batch的形式装data, 并且要to device且归一化
            embeddings = self.process_emb(embeddings)

            logger.info("Building index...")
            logger.info(f"Search length:{len(embeddings)}")
            self.index = {"sentences": sentences_or_file_path}
        
        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            if faiss_compress:
                nlist = 50  #聚类中心的个数 
                m = 8  # 压缩成8bits
                quantizer = faiss.IndexIVFPQ(quantizer, embeddings.shape[1],nlist,m,8)
                quantizer.nprobe = 10 #查找聚类中心的个数，默认为1个，若nprobe=nlist则等同于精确查找 
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path))) 
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    if n_gpus == 1:
                        res = faiss.StandardGpuResources()
                        res.setTempMemory(20 * 1024 * 1024 * 1024)
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    else:
                        ngpus = faiss.get_num_gpus()
                        print("number of GPUs:", ngpus)
                        index = faiss.index_cpu_to_all_gpus(index)
                else:
                    logger.info("Use CPU-version faiss")
            else: 
                logger.info("Use CPU-version faiss")

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            if faiss_compress:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def build_index(self, sentences_or_file_path: Union[str, List[str]], 
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        device: str = None,
                        batch_size: int = 64,
                        n_gpu: int = 2):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True 
            except:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}
        
        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path))) 
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    if n_gpu == 1:
                        res = faiss.StandardGpuResources()
                        res.setTempMemory(20 * 1024 * 1024 * 1024)
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    else:
                        ngpus = faiss.get_num_gpus()
                        print("number of GPUs:", ngpus)
                        index = faiss.index_cpu_to_all_gpus(index)
                else:
                    logger.info("Use CPU-version faiss")
            else: 
                logger.info("Use CPU-version faiss")

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def search_emb(self, queries,query_sents, 
                device: str = None, 
                threshold: float = 0.0,
                top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        # 如果想看一下intuition的话可以把topk搞小，然后利用query sents和results结合起来一起看
        query_vecs = queries
        query_vecs = query_vecs / np.linalg.norm(query_vecs,axis=1,keepdims=True)
        # query_vecs = query_vecs.to(device)

        distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
        
        def pack_single_result(dist, idx):
            results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
            chosen_idx = [i for i, s in zip(idx, dist) if s >= threshold]
            return results, np.array(chosen_idx)
        explore = False #用来控制是否要研究top2048的sample的pattern
        if len(queries) > 1:
            combined_results, chosen_idxs = [], []
            for i in range(len(queries)):
                results, chosen_idx = pack_single_result(distance[i], idx[i])
                combined_results.append(results)
                chosen_idxs.append(chosen_idx)
            chosen_idxs = np.concatenate(chosen_idxs)
            chosen_idxs = np.unique(chosen_idxs.flatten()) # 全部要就改成idx
            if explore:
                idx2counts = dict(Counter(chosen_idx.flatten()))
                idx2counts_sorted = sorted(idx2counts.items(), key=lambda item:item[1],reverse=True)
                topkk = 100
                topk_sents = [(self.index["sentences"][idx2counts_sorted[i][0]],idx2counts_sorted[i][1]/len(queries)) for i in range(topkk)]
                return combined_results, chosen_idxs, topk_sents
            print(f"We select {len(chosen_idxs)} from total Top-{top_k}*{len(combined_results)} samples.")
            return combined_results, chosen_idxs
        else:
            return pack_single_result(distance[0], idx[0])

    def search(self, queries: Union[str, List[str]], 
                device: str = None, 
                threshold: float = 0.6,
                top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results
            
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])

if __name__=="__main__":
    example_sentences = [
        'An animal is biting a persons finger.',
        'A woman is reading.',
        'A man is lifting weights in a garage.',
        'A man plays the violin.',
        'A man is eating food.',
        'A man plays the piano.',
        'A panda is climbing.',
        'A man plays a guitar.',
        'A woman is slicing a meat.',
        'A woman is taking a picture.'
    ]
    example_queries = [
        'A man is playing music.',
        'A woman is making a photo.'
    ]

    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    simcse = SimCSE(model_name)

    # print("\n=========Calculate cosine similarities between queries and sentences============\n")
    # similarities = simcse.similarity(example_queries, example_sentences)
    # print(similarities)

    # print("\n=========Naive brute force search============\n")
    # simcse.build_index(example_sentences, use_faiss=False)
    # results = simcse.search(example_queries)
    # for i, result in enumerate(results):
    #     print("Retrieval results for query: {}".format(example_queries[i]))
    #     for sentence, score in result:
    #         print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
    #     print("")
    
    print("\n=========Search with Faiss backend============\n")
    simcse.build_index(example_sentences, use_faiss=True)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")

