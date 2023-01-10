# Hausa VQA 
# prepare caption data

import os

import pandas as pd


def prepare_caption_data(fpath: str, save_path: str, trglang: str):
    """Prepare dataframe and save to file"""
    src_col_name = 2
    trg_col_name = 3
    df = pd.read_csv(fpath, encoding='utf-8', sep=',', header=None)
    print(df.head())
    df = df[[src_col_name, trg_col_name]]
    df.rename(columns={src_col_name: 'en', trg_col_name: trglang}, inplace=True)
    df.to_csv(save_path, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":

    for target_lang, full_lang, folder_name in zip(
            ['ha'],
            ['hausa'],
            ['HaVQA']
    ):
        inp_data_dir = os.path.join(f'../data/raw/{folder_name}')
        out_data_dir = os.path.join(f'../data/prepared/{folder_name}')

        hausa_vqa_data_paths = {
            'train': os.path.join(inp_data_dir, f'{full_lang}-train.txt'),
            'dev': os.path.join(inp_data_dir, f'{full_lang}-dev.txt'),
            'test': os.path.join(inp_data_dir, f'{full_lang}-test.txt')
        }

        # prepare caption data
        os.makedirs(out_data_dir, exist_ok=True)
        for split in ['train', 'dev', 'test']:
            prepare_caption_data(
                fpath=hausa_vqa_data_paths[split],
                save_path=os.path.join(out_data_dir, f'{split}.en.{target_lang}.tsv'),
                trglang=target_lang
            )
