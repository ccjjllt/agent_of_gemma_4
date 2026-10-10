"""Numerical two-rank CPU check of global-token weighting, padding and resume.

Run: /tmp/p005-training-check/bin/python -m training.tests.check_ddp
Uses random tiny Gemma4, five unequal examples, and actual DDP/Gloo collectives.
No models are downloaded and no GPU is used.
"""
import json
from pathlib import Path
import tempfile

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from training.memory import install_selected_token_loss
from training.tests.test_memory import tiny_lora
from training.train_ddp import run_ddp_loop
from training.train_sft import DEFAULTS, run_training_loop


ROWS = [
    {'input_ids':[1,4,5,6], 'attention_mask':[1]*4, 'labels':[-100,-100,5,6]},
    {'input_ids':[1,7,8], 'attention_mask':[1]*3, 'labels':[-100,-100,8]},
    {'input_ids':[1,9,10,11,12], 'attention_mask':[1]*5, 'labels':[-100,9,-100,11,12]},
    {'input_ids':[1,13,14,15], 'attention_mask':[1]*4, 'labels':[-100,-100,-100,15]},
    {'input_ids':[1,16,17], 'attention_mask':[1]*3, 'labels':[-100,16,17]},
]


def config():
    return dict(DEFAULTS,device='cpu',compute_dtype='float32',quant_backend='nf4',
        rank=2,alpha=4,epochs=1,micro_batch_size=1,gradient_accumulation_steps=2,
        learning_rate=.001,warmup_ratio=0.,checkpoint_every_n_steps=1,
        loss_backend='selected_chunked',loss_chunk_size=2,max_sequence_length=16)


def model():
    result=tiny_lora(checkpointing=True)
    install_selected_token_loss(result,chunk_size=2)
    return result


def rank_main(rank,root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(Path(root)/'rendezvous'))
    try:
        m=model()
        result=run_ddp_loop(m,ROWS,config(),Path(root)/'ddp',torch,0,'numeric')
        torch.save({n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad},Path(root)/('rank%d.pt'%rank))
        resumed=model()
        cp=Path(root)/'ddp/checkpoint-000001'
        set_peft_model_state_dict(resumed,load_file(cp/'adapter/adapter_model.safetensors'))
        again=run_ddp_loop(resumed,ROWS,config(),Path(root)/'resume',torch,0,'numeric',resume=cp)
        for (n,p),(other,q) in zip(m.named_parameters(),resumed.named_parameters()):
            if p.requires_grad:torch.testing.assert_close(p,q,rtol=0,atol=0,msg='DDP resume differs: '+n)
        (Path(root)/('rank%d.json'%rank)).write_text(json.dumps({'continuous':result,'resumed':again}))
    finally:
        torch.distributed.destroy_process_group()


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='p005-ddp-') as root:
        root=Path(root)
        for name in ['ddp','single','resume']:(root/name).mkdir()
        m=model();cfg=config();cfg['micro_batch_size']=2
        reference=run_training_loop(m,ROWS,cfg,root/'single',torch,0,'numeric')
        torch.multiprocessing.spawn(rank_main,args=(str(root),),nprocs=2,join=True)
        states=[torch.load(root/('rank%d.pt'%r),weights_only=True) for r in range(2)]
        maximum=0.
        for n,p in m.named_parameters():
            if not p.requires_grad:continue
            torch.testing.assert_close(states[0][n],states[1][n],rtol=0,atol=0)
            torch.testing.assert_close(p.detach(),states[0][n],rtol=1e-5,atol=2e-7,msg='Global objective differs: '+n)
            maximum=max(maximum,float((p.detach()-states[0][n]).abs().max()))
        result={'success':True,'world_size':2,'rows':5,'unequal_supervised_counts':True,
            'empty_last_rank_zero_weight':True,'adapter_max_abs_diff_vs_single':maximum,
            'ranks_bit_identical':True,'resume_bit_identical':True,'single':reference,
            'ddp':json.loads((root/'rank0.json').read_text())}
        print(json.dumps(result,indent=2))
        return result


if __name__=='__main__':main()
