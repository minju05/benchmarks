import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import re
import emoji
from tqdm import tqdm
from augment import EDA
from soynlp.normalizer import repeat_normalize

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DF_Preprocessor(object):
    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_data(cls, path):
        if path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    @classmethod
    def preprocess_df(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        self.df['comments'] = clean(str(self.df['comments']))
        self.df = pd.get_dummies(self.df, columns=['bias', 'hate'])
        return self.df

    def augmented(self, df):
        df = self.df ## hate label 위주로
        df = self.preprocess_df(df)
        df_list = df.values.tolist()
        result = []

        if self.args.augmented is False:
            return df

        for data in tqdm(df_list):
            left = data[:]

            # augmentation된 정보 담기
            augmented = EDA(data[0])

            for i, aug in enumerate(augmented):
                left[0] = aug
                # augmented된 텍스트의 경우 정보 입력
                if i != 6:
                    result.append(left + ['augmented_text'])
                # original 텍스트의 경우 정보 입력
                elif i == 6:
                    result.append(left + ['original_text'])

        result_df = pd.DataFrame(result, columns=list(df.columns) + ['type'])
        return result_df.to_pickle('augmented_df.csv')


class CommentsDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            max_token_len: int = 150
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    @property
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        label_columns = self.data.columns.tolist()[2:]
        comment_text = data_row.comments
        labels = data_row[label_columns]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class CommentDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = CommentsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = CommentsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )
