#!/usr/bin/env python3

precision = 15 # Количество итераций. Значение от 10 до 40. Чем меньше, тем быстрее считается. Чем больше, тем точнее считается. 
weighted_dict = {
                    ',': -0.073,
                    'и': -0.071,
                    'с': -0.025,
                    'а': -0.011,
                    'в': -0.010,
                    'президент': -0.009,
                    'также': -0.009,
                    'глава': -0.006,
                    'дело': -0.006,
                    'бывший': -0.006,
                    'иностранный': -0.005,
                    'что': -0.005,
                    'фонд': -0.005,
                    'встречаться': -0.004,
                    ')': -0.004,
                    'помощник': -0.004,
                    'встреча': -0.004,

                    'назначать': 0.004,
                    '"': 0.005,
                    'являться': 0.005,
                    'быть': 0.011,
                    'ведомство': 0.012,
                    'компания': 0.015,
                    }



import json
import sys
import numpy as np

from collections import defaultdict
from pymystem3 import Mystem
import matplotlib.pyplot as plt

from freparser.tokens import TokensStorage
from freparser.spans import SpansStorage
from freparser.objects import ObjectsStorage
from freparser.corefs import CorefsStorage
from freparser.facts import FactsStorage



def br():
    import pdb
    pdb.set_trace()

global_min = None
global_max = None
pos_thresholds = []
neg_thresholds = []

def simple_distance(segment, is_correct):
    global global_min
    global global_max
    global pos_thresholds
    global neg_thresholds
    res = -1 * len(segment)
    if global_min is None or res < global_min:
        global_min = res
    if global_max is None or res > global_max:
        global_max = res
    if is_correct:
        pos_thresholds.append(res)
    else:
        neg_thresholds.append(res)
    return res

def weighted_distance(segment, is_correct, word2weight, distance_base):
    global global_min
    global global_max
    global pos_thresholds
    global neg_thresholds
    res = -1 * distance_base * len(segment)
    for word in segment:
        #print(word)
        #print(word2weight.keys())
        if word in word2weight.keys():
            #print(111)
            res += word2weight[word]
    if global_min is None or res < global_min:
        global_min = res
    if global_max is None or res > global_max:
        global_max = res
    if is_correct:
        pos_thresholds.append(res)
    else:
        neg_thresholds.append(res)
    return res
    
m = Mystem()
cor_file = 'correct_answers'
mis_file = 'mistake_answers'
with open(cor_file, 'w') as f:
    pass
with open(mis_file, 'w') as f:
    pass

def lematize(word_list):
    words = word_list
    if type(words) == list:
        words = ' '.join(words)
    lemmas = m.lemmatize(words)
    return lemmas

def make_plot(data, file_name, x_label, y_label, step = None):
    plt.figure(1)
    plt.plot(data[0], data[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    xstep = step if step is not None else (max(data[0]) - min(data[0]))/50
    ax = plt.subplot(111)
    ax.set_xticks(np.linspace(min(data[0]), max(data[0]),
        50), minor=True
        )
    ax.grid(which='both')
    plt.savefig(file_name +  '.png')
    plt.clf()

class Main:
    def load_data(self, prefix):
        self.tokens = TokensStorage.load_from_file("{}.tokens".format(prefix))
        self.spans = SpansStorage.load_from_file("{}.spans".format(prefix), self.tokens)
        self.objects = ObjectsStorage.load_from_file("{}.objects".format(prefix), self.spans)
        self.corefs = CorefsStorage.load_from_file("{}.coref".format(prefix), self.objects)
        self.facts = FactsStorage.load_from_file("{}.facts".format(prefix))

    @classmethod
    def format_person(cls, coref):
        return " ".join(
            coref.props[key]
            for key in ["lastname", "firstname", "patronymic", "nickname"]
            if key in coref.props
        )
    
    @classmethod
    def format_occupation_type(cls, fact):
        types = []
        for code in ["Who", "Job", "Where"]:
            if code not in fact.props:
                continue
            id = fact.props[code][0]
            if fact.props[code][0].startswith("obj"):
                types.append(code + '=' + cls.corefs[int(id[3:])].objects[0].type)
            if fact.props[code][0].startswith("span"):
                types.append(code + '=' + cls.spans[int(id[4:])].type)
        return "|".join(types)
                
    
    @classmethod
    def get_entity(cls, code_name):
        if code_name.startswith('obj'):
            return cls.corefs[int(code_name[3:])]
        if code_name.startswith('span'):
            return cls.spans[int(code_name[4:])]
    
    @classmethod
    def get_fact_hash(self, fact):
        return 'Who={0}|Job={1}'.format(
                    fact.props['Who'][0], 
                    fact.props['Job'][0])
    
    def generate_fact_hash(self, person, job):
        pers_coref = self.corefs.get_by_object(person)
        return 'Who=obj{0}|Job=span{1}'.format(str(pers_coref.id), str(job.id))
    
    def get_correct_facts(self):
        if 'correct_hashes' not in dir(self):
            self.correct_hashes = set()
            correct_facts = self.facts.list_by_type("Occupation")
            for fact in correct_facts:
                try:
                    self.correct_hashes.add(self.get_fact_hash(fact))
                except:
                    pass
        return self.correct_hashes

    def check_response(self, response):
        correct_hashes = self.get_correct_facts()
        user_hashes = set(response)
        return {
            "correct": len(user_hashes.intersection(correct_hashes)),
            "false_positive": len(user_hashes - correct_hashes),
            "false_negative": len(correct_hashes - user_hashes),
        }

    def print_user_response(self, response):
        pass
        print('User response:')
        for i in response:
            print('\t' + str(i))
    
    def distance_function(self, segment):
        return len(segment)

    def simple_method(self):
        objs_by_sentence = defaultdict(list)
        for obj in self.objects.list_by_type("Person"):
            objs_by_sentence[obj.related_spans[0].tokens[0].sentence_idx].append(obj)
        return [
            sorted(set(
                self.corefs.get_by_object(obj).id
                for obj in objects
            ))
            for objects in objs_by_sentence.values()
            if len(objects) >= 2
        ]
    
    
    def tokens_between(self, pers, job, context_window = 0):
        pers_tokens = [tok.id for span in pers.related_spans for tok in span.tokens]
        job_tokens = [tok.id for tok in job.tokens]
        tokens = sorted(pers_tokens + job_tokens)
        mid = self.tokens.get_sentence(job.tokens[0].sentence_idx)
        mid = [token.text for token in mid if 
            token.id >= tokens[0] - context_window and 
            token.id <= tokens[-1] + context_window and 
            token.id not in tokens]
        return mid
    
    def print_segments_to_file(self, collection, file_name, context_window = 0):
        with open(file_name, 'a') as f:
            for pers,job in collection:
                f.write(' '.join(
                     lematize(
                     self.tokens_between(pers, job, context_window))))
        
        
        
    def simple_method2(self, weight_function=None):
        persons_by_sentence = defaultdict(list)
        for obj in self.objects.list_by_type("Person"):
            for span in obj.related_spans:
                persons_by_sentence[span.tokens[0].sentence_idx].append(obj)
        jobs_by_sentence = defaultdict(list)
        for span in self.spans.list_by_type("job"):
            jobs_by_sentence[span.tokens[0].sentence_idx].append(span)
        places_by_sentence = defaultdict(list)
        for obj in self.objects.list_by_type("Org") + self.objects.list_by_type("LocOrg"):
            for span in obj.related_spans:
                places_by_sentence[span.tokens[0].sentence_idx].append(obj)
        full_sents = set(persons_by_sentence.keys()).intersection(
                        set(jobs_by_sentence.keys()).intersection(
                            set(places_by_sentence.keys())))
        res = []
        
        pers_and_job = set(persons_by_sentence.keys()).intersection(set(jobs_by_sentence.keys()))
        job_and_place = set(places_by_sentence.keys()).intersection(set(jobs_by_sentence.keys()))
        
        job2pers = defaultdict(list)
        job2place = defaultdict(list)
        for sent_idx in pers_and_job:
            for job in jobs_by_sentence[sent_idx]:
                for pers in persons_by_sentence[sent_idx]:
                    job2pers[job].append(pers)
        for sent_idx in job_and_place:
            for job in jobs_by_sentence[sent_idx]:
                for place in places_by_sentence[sent_idx]:
                    job2place[job].append(self.corefs.get_by_object(place))
        res = set([])
        for job in job2pers.keys():
            for pers in job2pers[job]:
                for place in job2place[job]:
                    res.add((pers,job))
        
        for answer in res:
            is_correct = self.generate_fact_hash(answer[0],answer[1]) in self.get_correct_facts()
            #print(self.generate_fact_hash(answer[0],answer[1])
            #       + '-' + str(is_correct))
            if weight_function is not None:
                weight_function(self.tokens_between(answer[0], answer[1]), is_correct)
        
        correct_answers = [i for i in res if 
            self.generate_fact_hash(i[0], i[1]) 
            in self.get_correct_facts()]
        mistake_answers = [i for i in res if 
            self.generate_fact_hash(i[0], i[1]) 
            not in self.get_correct_facts()]
        self.print_segments_to_file(correct_answers, cor_file)
        self.print_segments_to_file(mistake_answers, mis_file)
        return [self.generate_fact_hash(pers,job) for pers, job in res
                    if weight_function(self.tokens_between(pers, job), True) >= -0.0217
                    ]

    def solve(self, weight_function=None):
        """ INSERT YOUR CODE HERE """
        a = self.simple_method2(weight_function)
        #print(a)
        return a

    @staticmethod
    def add_results(a, b):
        return {
            key: a[key] + b[key]
            for key in ["correct", "false_positive", "false_negative"]
        }

    @staticmethod
    def gen_results(correct, false_positive, false_negative):
        return {
            "correct": correct,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    @staticmethod
    def boost_result(result):
        precision = result["correct"] / (result["correct"] + result["false_positive"])
        recall = result["correct"] / (result["correct"] + result["false_negative"])
        try:
            fmeasure = 2 * (precision * recall) / (precision + recall)
        except:
            fmeasure = 0
        return {  
            "precision": precision,
            "recall": recall,
            "f-measure": fmeasure,
 #           **result,
        }

    @classmethod
    def run_single_file(cls, prefix, weight_function=None):
        instance = cls()
        instance.load_data(prefix)
        user_response = instance.solve(weight_function)
        #instance.print_user_response(user_response)
        #print(json.dumps(instance.check_response(user_response), ensure_ascii=False, indent=4, sort_keys=True))
        #print(json.dumps(self.simple_method(), ensure_ascii=False, indent=4, sort_keys=True))

    @classmethod
    def run_list(cls, filename, weight_function=None):
        result = {
            "correct": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        with open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                prefix = line.strip()
                instance = cls()
                instance.load_data(prefix)
                user_response = instance.solve(weight_function)
                current_result = instance.check_response(user_response)
                result = cls.add_results(result, current_result)
                #print("{:<20s} {correct:>3d} {false_positive:>3d} {false_negative:>3d}".format(prefix, **current_result))
        print(json.dumps(cls.boost_result(result), ensure_ascii=False, indent=4, sort_keys=True))
        return cls.boost_result(result)

    @classmethod
    def test_threshold(cls, filename, weight_function=None, method_name = 'null_method', plot = True):
        global global_min
        global global_max
        global pos_thresholds
        global neg_thresholds
        global_min = None
        global_max = None
        pos_thresholds = []
        neg_thresholds = []
        cls.run_list(filename, weight_function)
        res_list = []
        all_thresholds = sorted(list(set(neg_thresholds + pos_thresholds)))
        for threshold in all_thresholds:
            boost_result = cls.boost_result({
                "correct": len([i for i in pos_thresholds if i >= threshold]),
                "false_positive": len([i for i in neg_thresholds if i >= threshold]),
                "false_negative": len([i for i in pos_thresholds if i < threshold]),
                })
            fmeasure = boost_result["f-measure"]
            print(boost_result)
            res_list.append([threshold, fmeasure])
        res_list = np.array(res_list).T
        if plot:
            make_plot(res_list, method_name, 'Error threshold', 'F-measure')
        best_threshold = res_list[0, np.argmax(res_list[1])]
        best_fmeasure = res_list[1, np.argmax(res_list[1])]
        print("Optimal threshold = {0:.3}, fmeasure = {1:.3%}".format(
            best_threshold,
            best_fmeasure))
        return best_fmeasure
    

if __name__ == "__main__":
    if sys.argv[1] == "s":
        Main().run_single_file(sys.argv[2])
    elif sys.argv[1] == "l":
        weight_function = lambda segment, is_correct: weighted_distance(segment, is_correct, weighted_dict, 0.00172)
        Main().run_list(sys.argv[2], weight_function)
    elif sys.argv[1] == "t":
        #Main().test_threshold(sys.argv[2], simple_distance, 'simple_distance')
        fm = Main().test_threshold(sys.argv[2], simple_distance, 'simple_distance')
        with open('good_words.txt', 'r') as f:
            good_words = [i for i in lematize(f.read().split('\n')) if i[0] != ' ']
            #print(good_words)
        with open('bad_words.txt', 'r') as f:
            bad_words = [i for i in lematize(f.read().split('\n')) if i[0] != ' ']
        
        results = []
        for val in np.linspace(0, 0.01, precision):
                weighted_dict = weighted_dict
                func = lambda segment, is_correct: weighted_distance(segment, is_correct, weighted_dict, val)
                nm = Main().test_threshold(sys.argv[2], func, method_name='unidict, val=' + str(val), plot=True)
                results.append([val, nm])
        res_list = np.array(results).T
        best_val = res_list[0, np.argmax(res_list[1])]
        best_fmeasure = res_list[1, np.argmax(res_list[1])]
        print("Optimal distance base = {0:.3}, fmeasure = {1:.3%}".format(
        best_val,
        best_fmeasure))
        make_plot(res_list, 'distance2fmeasure', 'Distance base value', 'F-measure')
