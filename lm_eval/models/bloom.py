import transformers
import torch
from lm_eval.base import BaseLM
from accelerate.big_modeling import dispatch_model, infer_auto_device_map, get_balanced_memory


class BLOOMLM(BaseLM):

    def __init__(
        self,
        device="cuda",
        pretrained="bloom",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        is_fp16=False,
        load_8bit=False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            torch_dtype=torch.float16 if is_fp16 else torch.float32,
            load_in_8bit=load_8bit,
            device_map='auto' if load_8bit else None,
        )
        self.is_fp16 = is_fp16
        self.pretrained = pretrained
        self.no_split_modules = self.model._no_split_modules
        self.model.eval()
        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            # subfolder=subfolder,
            use_fast=True,
        )
        print(self.tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

    def prepare_for_inference(self):
        self.no_split_modules = self.model._no_split_modules
        if self.is_fp16:
            self.model.to(torch.float16)
        else:
            self.model.to(torch.float32)
        max_memory = get_balanced_memory(
            self.model,
            no_split_module_classes=self.no_split_modules,
            dtype=torch.float16 if self.is_fp16 else torch.float32,
        )
        device_map = infer_auto_device_map(
            self.model,
            no_split_module_classes=self.no_split_modules,
            dtype=torch.float16 if self.is_fp16 else torch.float32,
            max_memory=max_memory,
        )
        print(device_map)
        dispatch_model(self.model, device_map=device_map)
        self.model.eval()

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512
        # try:
        #     return self.model.config.n_ctx
        # except AttributeError:
        #     # gptneoconfig doesn't have n_ctx apparently
        #     return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, attention_mask=None):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps, attention_mask=attention_mask)[0][:, :, :250680]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
