import torch


def mlm_getitem(data, mlm_probability=0.15, contains_eos=False, tokenizer=None, eligible_replacements=torch.LongTensor([6,7,8,9,10]),
                special_tokens_mask=None, seed=0):
    """Helper method for creating MLM input / target.

    Adapted from:
    https://github.com/huggingface/transformers/blob/14666775a296a76c88e1aa686a9547f393d322e2/src/transformers/data/data_collator.py#L751
    """
    #torch.manual_seed(seed) #Added for context parallel to make reproducible

    #data = seq[:-1].clone() if contains_eos else seq.clone()  # remove eos, if applicable
    target = data.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(target.shape, mlm_probability)
    if special_tokens_mask is None:
        #special_tokens_mask = [
        #    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in target.tolist()
        #]
        special_tokens_mask = tokenizer.get_special_tokens_mask(target, already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = tokenizer.pad_token_id  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    data[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    if eligible_replacements is not None:
        rand_choice = torch.randint(eligible_replacements.shape[0], size=target.shape)
        random_words = eligible_replacements[rand_choice]
    else:
        random_words = torch.randint(len(tokenizer), size=target.shape, dtype=torch.long)
    data[indices_random] = random_words[indices_random]
    # FIXME this was breaking repro so commented it out
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return data, target

def mlm_esp_getitem(data, mlm_probability=0.15, seed=None):
    """Helper method for creating MLM input / target.
    for masking expression data
    """
    #torch.manual_seed(seed) #Added for context parallel to make reproducible
    target = data.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(target.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = -100  # We only compute loss on masked tokens
    ## 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    data[masked_indices] = -100
    #TODO 10% of the time shift the value
    return data, target
