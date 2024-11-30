class CrowdsourceChecker:
    def __init__(self, data_repository):
        self.data_repository = data_repository
        self.answer_dict = {}

        self.crowd_data = data_repository.crowd_data

        for batch_id, batch in self.crowd_data.items():
            for task_id, task in batch['tasks'].items():
                ent_rel_tuple = (task['Entity'], task['Relation'])
                self.answer_dict[ent_rel_tuple] = {}
                self.answer_dict[ent_rel_tuple]['Answer'] = task['Majority Element'] == 1.0
                self.answer_dict[ent_rel_tuple]['Support Votes'] = task['Votes'][0]
                self.answer_dict[ent_rel_tuple]['Reject Votes'] = task['Votes'][1]       
                self.answer_dict[ent_rel_tuple]['Kappa'] = batch['Kappa']           

    def exists_in_crowd_data(self, entity, relation):
        return ((entity, relation) in self.answer_dict)

    def get_interrater_agreement(self, entity, relation):
        return self.answer_dict[(entity, relation)]['Kappa']

    def get_votes(self, entity, relation):
        return self.answer_dict[(entity, relation)]['Reject Votes'], self.answer_dict[(entity, relation)]['Support Votes']