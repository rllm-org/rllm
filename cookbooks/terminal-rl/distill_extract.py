"""Extract thinking-SFT rows from a terminus2 eval run's episode dumps.

Stock ``rllm dataset from-eval`` drops ``reasoning_content`` (curation
``_clean_message``) and reads only the final step's conversation, where the
harness has already stripped historical reasoning. For distilling a thinking
model that is unusable, so this extractor rebuilds the full