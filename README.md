# Topic Modeling on Twitter data

## Getting started
- install the necessary dependencies using `pip istall -r requrements.txt`
- run the topic modeling with the desired modeling technique:
    - `python topic_modeling.py --model bert`
    - `python topic_modeling.py --model lda`


## Approach
- Cleaning the data
- Ideally the next step should be lemmatization or stemming, however for hebrew the few ones I found eitehr didn't run (needs more debugging time) or were LLMs that are too big for my computer to run on.
- Run LDA or BERTtopic
- LDA (and other mthods like HDBSCAN) don't output a readable topic but rather keywords that belong to a topic - so my idea was to still use them but add a generative LLM after that, which will create this readable topic out of the keywords. With a bit of prompt enngineering (like also giving the relevant tweets where the topic appears) this can get nice readable results. Larger models did not work on my coputer so I had to use gpt-2 which is inherently not good at foreign languages anyways so it doesn not give good results. Testing it on ChatGPT resulted in promising topics.


## Language Challenges
- finding lemmatizers and stemmers - there is not a lot available
- Hebrew is a language where there are no vowels written so the meaning of the word depends on the context, which makes it especially for classical techniques quite challenging
- Hebrew also has abbreviations written as '' which have to be handled - I did not do that within the scope of this exercise
- Hebrew is also written from right to left, which has an impact on some processing techniques and should be kept in mind
- In another run I would also remove the years in the dataset 


## Result:
- The missing lemmatization is especially visible in the BERTopic approach (result in topics consisting of the same word sematically)
- The diversity score of the LDA technique is a bit better, but that probably comes from the few topics in BERTopic that only contain variations of the same word - given the nature of Hebrew I expect models with embeddings to work better
