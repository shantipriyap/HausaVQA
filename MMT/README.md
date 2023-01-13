# HaVQA MMT  

The repository contains the code/data for the below tasks.
* English->Hausa Multimodal Translation 

## Assumption
The object tags were already extracted using Faster-RCNN and added to the repository for multimodal machine translation (MMT).

## Abbreviations Used 
* Hausa-- ha
* English --en 


## Environment Details

* Pytorch
* sentencepiece
* sacrebleu
* transformers

Note: Based on you environment (e.g. CUDA) select the Pytorch version

Example: 
* Name: torch
* Version: 1.7.1+cu110
* Name: transformers                          
* Version: 4.6.0
* Name: sentencepiece                         
* Version: 0.1.96
* Name: sacrebleu                             
* Version: 1.5.1


## Multimodal Tasks 

```
$ cd src
```

Finetune M2M-100 for multimodal translation

```
$ python finetune_mbart.py ha multimodal ../data/prepared_object_tags
```
