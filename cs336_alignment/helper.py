import torch
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # First, encode all sequences to determine max length
    encoded_sequences = []
    output_lengths = []
    
    # Get special tokens from the tokenizer
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    for i, (prompt, output) in enumerate(zip(prompt_strs, output_strs)):
        # Encode prompt and output separately
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        
        # Store the sequence and output length
        if i == 0 and bos_token_id is not None:
            # For the first sequence, add BOS token to the prompt if it exists
            encoded_sequences.append([bos_token_id] + prompt_ids + output_ids)
        else:
            encoded_sequences.append(prompt_ids + output_ids)
        output_lengths.append(len(output_ids))
    
    # Find max length - only add EOS to first sequence, then pad all to same length
    max_length = max(len(seq) for seq in encoded_sequences)
    if len(encoded_sequences[0]) == max_length:
        # Add EOS token to first sequence if it's the longest
        encoded_sequences[0] = encoded_sequences[0] + [eos_token_id]
        max_length += 1
    
    # Now process each sequence
    input_ids = []
    labels = []
    response_mask = []
    
    for i, (ids, output_len) in enumerate(zip(encoded_sequences, output_lengths)):
        # Calculate padding needed
        padding_len = max_length - len(ids)
        
        # Create padded sequence
        padded_seq = ids + [pad_token_id] * padding_len
        
        # For input_ids, take all but the last token
        input_ids.append(padded_seq[:-1])
        
        # For labels, take all but the first token
        labels.append(padded_seq[1:])
        
        # Create response mask
        # The mask should be 0 for prompt tokens and padding, 1 for output tokens
        prompt_len = len(ids) - output_len
        mask = [0] * prompt_len + [1] * output_len + [0] * padding_len
        # Slice mask to match input_ids/labels length (labels perspective: shift by 1)
        response_mask.append(mask[1:])
    
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor(response_mask)
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: shape [..., num_classes], raw logits (not probabilities)
    # Compute the log-probabilities
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - lse  # shape [..., num_classes]
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # log_probs: Shape (batch_size, sequence_length, vocab_size)
    # Contains log probabilities for every token in the vocabulary at each position

    # labels: Shape (batch_size, sequence_length)
    # Contains the actual token IDs that should have been predicted

    # labels.unsqueeze(-1): Shape (batch_size, sequence_length, 1)
    # Adds a dimension at the end to match torch.gather's requirements

    # torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)):
    # dim=-1 means we're gathering along the vocabulary dimension
    # For each position, it selects the log probability corresponding to the actual token ID
    # Output shape: (batch_size, sequence_length, 1)

    # .squeeze(-1): Shape (batch_size, sequence_length)
    # Removes that extra dimension we added
    response_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": response_log_probs, "token_entropy": token_entropy}
    else:
        return {"log_probs": response_log_probs}