from .constructor import Constructor


class DataConstructor(Constructor):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        
    def make_dataloader(self):
        dataprocessor = self.construct_class(self.config_dict['data_processor'],
                                             'data_processor')
        
        dataset = self.construct_class(self.config_dict['dataset_reader'], 
                                       'dataset_reader')
        
        if "allennlp" in str(self.config_dict['data_loader']):
            data_loader = self.construct_class(self.config_dict['data_loader'], 
                                               'data_loader', 
                                               reader=dataset)
        else:
            data_loader = self.construct_class(self.config_dict['data_loader'], 
                                               'data_loader', 
                                               dataset=dataset,
                                               collate_fn=dataprocessor.collate_fn)
        for idx, b in enumerate(data_loader):
            print(idx, b)
            list(b)
            break

        if "allennlp" in str(self.config_dict['data_loader']):
            vocab = self.construct_class(self.config_dict['vocabulary'],"vocabulary")
            data_loader.index_with(vocab)
                     
        # 패키지 별로 data loading format 동일하게 후처리 wrapping 해주기
        """
        {'x': 
            {
                'x': 
                {
                    'token_ids': tensor([[  101,  1523,  1037,  ...,     0,     0,     0],
                                        [  101,  2045,  2024,  ...,     0,     0,     0],
                                        [  101, 16674,  3396,  ...,  1012,   102,     0]]), 
                    'mask': tensor([[ True,  True,  True,  ..., False, False, False],
                                        [ True,  True,  True,  ..., False, False, False],
                                        [ True,  True,  True,  ...,  True,  True, False]]), '
                    type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
                                        [0, 0, 0,  ..., 0, 0, 0],
                                        [0, 0, 0,  ..., 0, 0, 0]])
                }
            }, 
        'labels': tensor([[0, 1, 0,  ..., 0, 0, 0],
                        [1, 0, 0,  ..., 0, 0, 0],
                        [0, 0, 0,  ..., 0, 0, 0]]), 
        'meta': [
            {
                'title': 'Wilderness', 
                'body': '“A novel of violence, crisp dialogue, and suspense. . . . The reader is immediately caught up in the ambience of danger.”—The Boston GlobeAt forty-six, Aaron Newman was enjoying the good things in life—a good marriage, a good job—and he was in good shape himself. Then he saw the murder. A petty vicious killing that was to plunge him into an insane jungle of raw violence and fear, threatening and defiling the things he cared about.', 
                'topics': {'d2': ['Noir Mysteries', 'Crime Mysteries', 'Suspense & Thriller'], 'd0': 'Fiction', 'd1': 'Mystery & Suspense'}, 
                'idx': '9780440193289', 
                'using_tc': False}, 
            {
                'title': '940 Saturdays', 
                'body': 'There are 940 Saturdays between a child’s birth and the day he or she turns 18. That may sound like a lot when there are adventures to plan and hours to fill. But as your child learns to walk, ride a bicycle, and drive, the years pass quickly. This beautiful package includes both a removable booklet with a thousand ideas for family activities that you and your child will love at every age, and a keepsake journal for preserving what you saw and did, thought and felt, so you can savor these memories in the years to come.', 
                'topics': {'d0': 'Nonfiction', 'd1': 'Parenting'}, 
                'idx': '9780804185424', 
                'using_tc': False}, 
            {
                'title': 'And Our Faces, My Heart, Brief as Photos', 
                'body': 'Booker Prize-winning author John Berger reveals the ties between love and absence, the ways poetry endows language with the assurance of prayer, and the tensions between the forward movement of sexuality and the steady backward tug of time. He recreates the mysterious forces at work in a Rembrandt painting, transcribes the sensorial experience of viewing lilacs at dusk, and explores the meaning of home to early man and to the hundreds of thousands of displaced people in our cities today. And Our Faces, My Heart, Brief as Photos is a seamless fusion of the political and personal.', 
                'topics': {'d0': ['Classics', 'Poetry'], 'd1': 'Literary Criticism'}, 
                'idx': '9780679736561', 
                'using_tc': False}, 

        """
        return data_loader, vocab


