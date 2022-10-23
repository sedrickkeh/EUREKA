def truncate(sent):
    # sent_clean = sent.replace(" @", '')
    sent_clean = sent
    sent_split = sent_clean.split()
    start_idx, end_idx = -1, -1
    for i in range(len(sent_split)):
        if "<" in sent_split[i]: start_idx=i
        if ">" in sent_split[i]: end_idx=i
    start_idx = max(start_idx-10, 0)
    end_idx = min(end_idx+10, len(sent_split))
    sent_new = sent_split[start_idx:end_idx]
    sent_new = ' '.join(sent_new)
    return sent_new

def clean(sent):
    sent_clean = sent.replace(" @", '')
    return sent_clean
