"""Nucleotide Transformer Benchmarks Dataset.

From: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
"""

import torch
from datasets import load_dataset
import pandas as pd
import anndata as ad
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement
#from caduceus.tokenization_caduceus import CaduceusTokenizer

class ReplogleDataset(torch.utils.data.Dataset):
    """
    Loop through fasta file for sequence.
    Returns a generator that retrieves the sequence.
    """

    def __init__(
            self,
            split,
            # max_length,
            expression_h5ad="data/replogle/K562_gwps_raw_bulk_01.h5ad",
            d_output=2,  # default binary classification
            tokenizer=None, #CaduceusTokenizer(model_max_length=MAX_ALLOWED_LENGTH),
            tokenizer_name=None,
            use_padding=None,
            add_eos=True,
            add_cls=True,
            rc_aug=False,
            conjoin_train=False,
            conjoin_test=False,
            return_augs=False,
            shuffle_genes=True,
    ):

        # self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.d_output = d_output  # needed for decoder to grab
        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.shuffle_genes = shuffle_genes
        self.split = split

        # For NT tasks, we use data from InstaDeepAI/nucleotide_transformer_downstream_tasks
        self.seqs = torch.load('data/replogle/coding_by_gene.pt')
        self.samples = ad.read_h5ad(expression_h5ad)
        self.target_genes = self.samples.var.index.tolist()
        self.num_targets = len(self.target_genes)
        self.sample_genes = self.samples.obs.index.tolist()
        self.num_samples = len(self.sample_genes)
        self.sgrnas = pd.read_excel("data/replogle/media-2.xlsx", index_col='unique sgRNA pair ID')
        # print(self.sgrnas)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        target_name = self.sample_genes[idx]
        sgrna = self.sgrnas.loc[target_name]
        sgrna_seqs = [sgrna['targeting sequence A'], sgrna['targeting sequence B']]
        target_expr = self.samples[target_name].X.squeeze()

        indices = (torch.randperm(self.num_targets) if self.shuffle_genes else torch.arange(self.num_targets)).tolist()

        seq_ids, y = [], []
        for x in sgrna_seqs:
            seq = self.tokenizer(
                x,
                add_special_tokens=False,
                padding='do_not_pad',  # "max_length" if self.use_padding else None,
                max_length=None,  # self.max_length,
                truncation=False,  # True,
            )
            seq_ids += seq["input_ids"]
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_ids.append(self.tokenizer.sep_token_id)

                # TODO need to add all sample genes to this list too
        for idx in indices:
            gene = self.target_genes[idx]
            x = "".join(self.seqs[gene])
            if len(x) == 0:
                # print("no gene for ", gene)
                continue
            y.append(target_expr[idx].item())
            # I don't know what this is but it was here before
            if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
                x = string_reverse_complement(x)
            # print('idx',idx,len(x))
            seq = self.tokenizer(
                x,
                add_special_tokens=False,
                padding='do_not_pad',  # "max_length" if self.use_padding else None,
                max_length=None,  # self.max_length,
                truncation=False,  # True,
            )
            seq_ids += seq["input_ids"]  # get input_ids
            if self.add_cls:
                seq_ids.append(self.tokenizer.cls_token_id)
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_ids.append(self.tokenizer.sep_token_id)
        # print(seq_ids)
        seq_ids = torch.LongTensor(seq_ids)
        # print(y.shape)
        target = torch.Tensor(y)
        # target = torch.from_numpy(np.concat(y))
        return seq_ids, target
