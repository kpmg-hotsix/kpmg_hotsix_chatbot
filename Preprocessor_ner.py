class Preprocessor:

    def __init__(self, ) :
        self.mapping = {
            0: 0,      
            1: 1,       
            -100: -100, # LABEL_PAD_TOKEN
        }

    def __call__(self, datasets) :
        ner_tags = datasets["labels"]
        batch_size = len(ner_tags)

        p_tags = []
        for i in range(batch_size):
            ner_tag = ner_tags[i]

            tag = [self.mapping[t] for t in ner_tag]
            p_tags.append(tag)

        datasets["labels"] = p_tags
        return datasets
