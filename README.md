# Survey Branch - LGBTQ+ Bias in LLM Prompts

## Survey

Collect paired creative writing prompts to measure whether language models generate biased or stereotypical content when romance protagonists are LGBTQ+ versus cisgender heterosexual. Goal: build a benchmark dataset for quantifying homophobia/transphobia in AI outputs.

[link to the survey](https://qualtricsxm2c76j77g8.qualtrics.com/jfe/form/SV_56JKqTNSCSGqMlM)

## TODO:

- move all datasets to dataset folder and all results to results folder
- add evaluation code to src, continue refactoring (maybe move masks.py functions to Dataset class when broadening it for Ibrahim's dataset)
- also Ibrahim wants to add another mask for non-binary people
- make src.gen prettier
- distribute more the survey (literature students ? also, Ibrahim, to distribute it on a Discord server would be great but I'm not on an LGBTQ one and if I was to enter and immediately spam...)
- augment the survey dataset with TIDAL
- add preemptive sentiment analysis
- reserve a gpu on g5k
- add identification on results data structure for lgbtq prompt vs cishet
- add string in save file to register model's temperature and top_p (btw, I think 0.95 top_p is fine across, but I don't remember quite well how temperature works and maybe it would be worth running a test with it at the highest)
- prepare script chatgpt 5 tokens from Muhammad 


