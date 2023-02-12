from tasks.data_utils import InputExample


class PVP:
    def __init__(self, tokenizer, max_src_length, max_tgt_length, task_mask=False):
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.task_mask = task_mask

    @property
    def cls_id(self):
        return self.tokenizer.get_command('ENC').Id

    @property
    def mask_id(self):
        return self.tokenizer.get_command('MASK').Id

    def encode(self, example: InputExample):
        raise NotImplementedError


class P3PVP(PVP):
    @property
    def mask_id(self):
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        return self.tokenizer.get_command(mask_token).Id

    def encode(self, example: InputExample):
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        if len(source_tokens) > self.max_src_length - 2:
            source_tokens = source_tokens[:(self.max_src_length - 2)]
        source_tokens = [self.cls_id] + source_tokens + [self.mask_id]
        return source_tokens, target_text