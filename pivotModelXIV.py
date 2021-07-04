from collections import defaultdict
from LandScape import LandScape
import numpy as np
from tools import *
from itertools import combinations
import random

import sys

import time

class Industry:
    """
    what classical NK models fail to capture:
        1. first -> whether include or exclude an element
        2. second -> how to configure this element
    so we want to design a 3-state NK model
        1. 2 -> not included
        2. 0 -> first state
        3. 1 -> second state
    """

    def __init__(self, N, K, K_overlap):

        self.N = N
        self.K = K
        self.K_overlap = K_overlap
        self.first_IM = None
        self.first_IM_dic = None
        self.FC = None
        self.FC_base = None
        self.cache = {}

    def create_influence_matrix(self):
        IM = np.eye(self.N)

        for i in range(self.N):
            probs = [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
            ids = np.random.choice(self.N, self.K, p=probs, replace=False)
            for index in ids:
                IM[i][index] = 1

        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)
        self.first_IM, self.first_IM_dic = IM, IM_dic

    def create_fitness_config(self, ):
        FC = defaultdict(dict)

        for row in range(self.N):

            first_dependency = [cur for cur in self.first_IM_dic[row] if cur != row]

            base = 2 * pow(3, len(first_dependency))

            for i in range(base):
                FC[row][i] = np.random.uniform(-0.5, 0.5)

        self.FC = FC

    def calculate_fitness(self, state):
        res = 0.0
        count = 0
        for i in range(len(state)):

            if state[i] == "2":
                continue

            first_dependency = self.first_IM_dic[i]

            index = 0

            base = 1

            for cur in [j for j in range(self.N - 1, -1, -1) if j != i] + [i]:

                if cur not in first_dependency:
                    pass

                if cur == i:

                    if state[cur] == "1":
                        index = index + base * 1
                    else:
                        index = index + base * 0

                elif cur in first_dependency:

                    if state[cur] == "2":
                        index = index + base * 0
                    elif state[cur] == "0":
                        index = index + base * 1
                    elif state[cur] == "1":
                        index = index + base * 2
                    base = base * 3

            res += self.FC[i][index]
            count += 1

        return res

    def store_cache(self, ):
        for i in range(pow(3, self.N)):

            ternary_string = toStr(i, 3)

            if len(ternary_string) < self.N:
                ternary_string = "0" * (self.N - len(ternary_string)) + ternary_string

            self.cache[ternary_string] = self.calculate_fitness(ternary_string)

    def query_fitness(self, state, decision):

        ternary_string = ""

        for i in range(len(state)):

            if i not in decision:
                ternary_string += "2"

            else:
                ternary_string += str(state[i])

        return self.cache[ternary_string]

    def query_sub_fitness(self, state, decision,):

        res = 0

        for i in range(len(state)):

            if i not in decision:
                continue

            if state[i] == "2":
                continue

            first_dependency = self.first_IM_dic[i]

            index = 0

            base = 1

            for cur in [j for j in range(self.N - 1, -1, -1) if j != i] + [i]:

                if cur not in first_dependency:
                    pass

                if cur == i:

                    if state[cur] == "1":
                        index = index + base * 1
                    else:
                        index = index + base * 0

                elif cur in first_dependency:

                    if state[cur] == "2":
                        index = index + base * 0
                    elif state[cur] == "0":
                        index = index + base * 1
                    elif state[cur] == "1":
                        index = index + base * 2
                    base = base * 3

            res += self.FC[i][index]

        return res

    def initialize(self, first_time=True, norm=True, change_influence=False):
        if first_time:
            self.create_influence_matrix()

        if change_influence:
            self.create_influence_matrix()

        self.create_fitness_config()
        self.store_cache()

        # normalization
        if norm:
            normalizer = max(self.cache.values())
            min_normalizer = min(self.cache.values())

            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - min_normalizer) / (normalizer - min_normalizer) - 0.5


class CEO:

    def __init__(self, managers, capacity, state, decision, industry):

        self.managers = managers
        self.capacity = capacity
        self.state = state
        self.decision = decision
        self.industry = industry
        self.hold_decision = []

        # if ceo decide to do pivot
        # delete is ok -> since deleted decision has a parent manager
        # what about added

    def integration(self, proposals, decentralized=False):

        base = 1
        for proposal in proposals:
            base*=len(proposal)

        remain_capacity = min(self.capacity, base)

        if decentralized:

            choices = [0]

        else:

            choices = np.random.choice(base, remain_capacity, replace=False)

        pool = []

        for choice in choices:

            choice = choice

            selected_proposal = []

            inner_base = int(base)

            for cur in range(len(proposals)):

                inner_base = inner_base//len(proposals[cur])
                index = choice // inner_base
                selected_proposal.append(proposals[cur][index])

                choice = choice % inner_base

            temp_state = list(self.state)

            for manager_index, manager in enumerate(self.managers):

                for d in manager.decision:

                    temp_state[d] = selected_proposal[manager_index][d]

            pool.append((temp_state, self.industry.query_fitness(temp_state, self.decision)))
        pool.sort(key=lambda x: -x[1])

        if decentralized:
            self.state = list(pool[0][0])
        else:
            if pool[0][1] > self.industry.query_fitness(self.state, self.decision):
                self.state = list(pool[0][0])

    def distribute_capacity(self, even=True, weighted=False, included_last=True):

        overall_sub_capacity = sum([manager.capacity for manager in self.managers])

        if included_last:
            manager_number = len(self.managers)
        else:
            manager_number = len(self.managers) - 1

        if even:
            capacity_list = [overall_sub_capacity//manager_number] * manager_number

            while sum(capacity_list) < overall_sub_capacity:

                choice = np.random.choice(manager_number)

                capacity_list[choice] = capacity_list[choice] + 1

        elif weighted:
            decision_list = [manager.decision for manager in self.managers[:manager_number]]
            weights = [decision_num / sum(decision_list) for decision_num in decision_list]
            capacity_list = [int(overall_sub_capacity * weight) for weight in weights]

            while sum(capacity_list) < overall_sub_capacity:

                choice = np.random.choice(manager_number)
                capacity_list[choice] = capacity_list[choice] + 1

        for cur in range(manager_number):
            self.managers[cur].capacity = capacity_list[cur]

    def adjust_pivot_decision(self):

        if len(self.hold_decision)==0:
            return

        choice = np.random.choice(self.hold_decision)

        temp_state = list(self.state)
        temp_state[choice] = temp_state[choice] ^ 1

        if self.industry.query_fitness(temp_state, self.decision) > self.industry.query_fitness(self.state, self.decision):
            self.state = list(temp_state)

    def reassign_controled_decision(self, random=True, even=False, relevant=False):

        if len(self.hold_decision)==0:
            return

        for d in self.hold_decision:

            if random:

                choice = np.random.choice(len(self.managers))
                self.managers[choice].decision.append(d)

            elif even:

                current_decision = [(cur, len(self.managers[cur].decision)) for cur in range(len(self.managers))]
                current_decision.sort(key=lambda x: x[1])

                self.managers[current_decision[0][0]].decision.append(d)

            elif relevant:

                interdependency = self.industry.first_IM_dic[d]

                current_interdependency = [
                    (cur, sum(
                        [sub_d in self.managers[cur].decision for sub_d in interdependency]
                    )) for cur in range(len(self.managers))
                ]

                current_interdependency.sort(key=lambda x: -x[1])
                self.managers[current_interdependency[0][0]].decision.append(d)
        self.hold_decision=[]

    def top_pivot(self, random=True, even=False, relevant=False, control=False):

        choice = np.random.choice(len(self.state))

        # would CEO consider a bottom line scale for each department ?
        # first let's assume yes

        for cur in range(len(self.managers)):

            if choice in self.managers[cur].decision and len(self.managers[cur].decision) <= self.managers[cur].constrain:
                return

        if choice in self.decision:
            index = self.decision.index(choice)
            temp_decision = list(self.decision)
            temp_decision.pop(index)

            if self.industry.query_fitness(
                self.state, temp_decision
            ) > self.industry.query_fitness(
                self.state, self.decision
            ):
                self.decision = list(temp_decision)

                for cur in range(len(self.managers)):

                    if choice in self.managers[cur].decision:

                        index = self.managers[cur].decision.index(choice)
                        self.managers[cur].decision.pop(index)

                if len(self.hold_decision) > 0:

                    if choice in self.hold_decision:
                        index = self.hold_decision.index(choice)
                        self.hold_decision.pop(index)
        else:

            temp_decision = list(self.decision)
            temp_decision.append(choice)

            temp_state = list(self.state)
            temp_state[choice] = np.random.choice([0, 1])

            if self.industry.query_fitness(
                temp_state, temp_decision
            ) > self.industry.query_fitness(
                self.state, self.decision
            ):
                self.decision = list(temp_decision)
                self.state = list(temp_state)

                if random:
                    # random

                    manager_index = np.random.choice(len(self.managers))
                    self.managers[manager_index].decision.append(choice)

                elif relevant:
                    # most relevant one

                    interdependency = self.industry.first_IM_dic[choice]

                    current_interdependency = [
                        (cur, sum(
                            [sub_d in self.managers[cur].decision for sub_d in interdependency]
                        )) for cur in range(len(self.managers))
                    ]

                    current_interdependency.sort(key=lambda x: -x[1])
                    self.managers[current_interdependency[0][0]].decision.append(choice)

                elif even:

                    current_decision = [(cur, len(self.managers[cur].decision)) for cur in range(len(self.managers))]
                    current_decision.sort(key=lambda x: x[1])

                    self.managers[current_decision[0][0]].decision.append(choice)
                elif control:
                    # direct control from CEO

                    self.hold_decision.append(choice)

                    # what about independent

class Manager:

    def __init__(self, decision, state, industry, capacity, partner, ceo, constrain):
        self.decision = decision
        self.state = state
        self.industry = industry
        self.capacity = capacity

        # partner is only used in decentralized search
        self.partner = partner
        self.ceo = ceo
        self.constrain = constrain

    def inner_layer_proposal(self, layer, remain_capacity):

        if remain_capacity==0:
            return []

        pool = []

        all_alternatives = list(combinations([cur for cur in range(len(self.decision))], layer))
        random.shuffle(all_alternatives)

        for alternative in all_alternatives:
            temp_state = list(self.state)

            for cur in alternative:
                temp_state[self.decision[cur]] = temp_state[self.decision[cur]] ^ 1

            pool.append(list(temp_state))

        if len(all_alternatives) < remain_capacity:

            if layer==len(self.decision):
                # exhaustive search
                return pool
            return pool + self.inner_layer_proposal(layer+1, remain_capacity-len(pool))

        else:
            choices = np.random.choice(len(pool), remain_capacity, replace=False)

            return [pool[c] for c in choices]

    def submit_proposal(self,):

        alternatives = [list(self.state)]+self.inner_layer_proposal(1, int(self.capacity)-1)

        ranked_alternative = [
            (
                alternatives[cur], self.industry.query_sub_fitness(
                    alternatives[cur], self.decision,
                )
            ) for cur in range(len(alternatives))
        ]

        ranked_alternative.sort(key=lambda x: -x[1])

        return [ranked_alternative[cur][0] for cur in range(len(ranked_alternative))]

    def change_decision(self, ):

        changeable_decision = [cur for cur in range(len(self.state)) if cur not in self.partner.decision]

        choice = np.random.choice(changeable_decision)

        if choice in self.decision and len(self.decision) <= self.constrain:

            return

        temp_decision = list(self.decision)

        if choice in self.decision:

            index = temp_decision.index(choice)
            temp_decision.pop(index)

            if self.industry.query_sub_fitness(
                    self.state, temp_decision
            ) > self.industry.query_sub_fitness(self.state, self.decision):
                self.decision = temp_decision
        else:
            temp_decision.append(choice)

            temp_state = list(self.state)
            temp_state[choice] = np.random.choice([0, 1])

            if self.industry.query_sub_fitness(
                temp_state, temp_decision
            ) > self.industry.query_sub_fitness(self.state, self.decision):
                self.decision = temp_decision
                self.state = temp_state

# initial elements
initial_elements = 8
change_interval = 10


class Firm:

    def __init__(self, N, altceo, altsub, industry, constrain=1):

        # initially we set to to 8 decision and 4 for each manager

        self.N = N
        self.industry = industry
        self.ceo = CEO(None, altceo,
            np.random.choice([0, 1], N).tolist(), np.random.choice(self.N, initial_elements, replace=False).tolist(),industry
        )
        self.manager_a = Manager(self.ceo.decision[:4], self.ceo.state, industry, altsub, None, self.ceo, constrain)
        self.manager_b = Manager(self.ceo.decision[4:], self.ceo.state, industry, altsub, None, self.ceo, constrain)

        self.ceo.managers = [self.manager_a, self.manager_b]
        self.manager_a.partner = self.manager_b
        self.manager_b.partner = self.manager_a

        self.fitness_values = self.industry.query_fitness(self.ceo.state, self.ceo.decision)

    def decentralized_configure(self):

        proposals = []

        for manager in self.ceo.managers:

            proposals.append(manager.submit_proposal())

        self.ceo.integration(proposals, decentralized=True)

        for manager in self.ceo.managers:
            manager.state = self.ceo.state


    def centralized_configuration(self, pivot_under_control=False):

        proposals = []

        for manager in self.ceo.managers:
            proposals.append(manager.submit_proposal())

        self.ceo.integration(proposals, decentralized=False)

        if pivot_under_control:
            self.ceo.adjust_pivot_decision()

        for manager in self.ceo.managers:
            manager.state = self.ceo.state

    def centralized_pivot(self, existing, under_control, independent):

        # existing business
        if existing is not None:

            if existing==0:
                # random
                self.ceo.top_pivot(random=True, even=False, relevant=False, control=False)
            elif existing==1:
                # even
                self.ceo.top_pivot(random=False, even=True, relevant=False, control=False)
            elif existing==2:
                # relevant
                self.ceo.top_pivot(random=False, even=False, relevant=True, control=True)

        # under control

        if under_control is not None:
            self.ceo.top_pivot(random=False, even=False, relevant=False, control=True)

        if independent is not None:
            self.ceo.top_pivot(random=False, even=False, relevant=False, control=True)

    def decentralized_pivot(self):

        for manager in self.ceo.managers:

            manager.change_decision()
            self.ceo.state = manager.state

            for inner_manager in self.ceo.managers:

                inner_manager.state = self.ceo.state
        self.ceo.decision = []
        for manager in self.ceo.managers:
            self.ceo.decision += manager.decision

    def adaptation(self, step, decentralized_configuration=True, decentralized_pivot=True, merge=True,
                   merge_assign=0, under_control=True, control_assign=0,independent=False, independent_keep=False,
                   independent_assign=0, limited_capacity=False, capacity_assign=0):

        if decentralized_configuration:
            if decentralized_pivot:
                # decentralized configuration & decentralized pivot
                if (step//change_interval) % 2 == 0:
                    self.decentralized_configure()
                else:
                    self.decentralized_pivot()

                    if limited_capacity:
                        if ((step+1)//change_interval) % 2 ==0:
                            if capacity_assign==0:
                                self.ceo.distribute_capacity(even=True, weighted=False, included_last=True)
                            else:
                                self.ceo.distribute_capacity(even=False, weighted=True, included_last=True)
            else:

                if merge:
                    if (step // change_interval) % 2 == 0:
                        self.decentralized_configure()
                    else:
                        self.centralized_pivot(existing=merge_assign, under_control=None, independent=None)

                        if limited_capacity:
                            if ((step + 1) // change_interval) % 2 == 0:
                                if capacity_assign == 0:
                                    self.ceo.distribute_capacity(even=True, weighted=False, included_last=True)
                                else:
                                    self.ceo.distribute_capacity(even=False, weighted=True, included_last=True)

                elif under_control:

                    # decentralized configuration & centralized pivot (under control random assign)
                    if (step//change_interval) % 2 ==0:
                        self.decentralized_configure()
                        self.ceo.adjust_pivot_decision()
                        if ((step+change_interval//2)//change_interval) % 2==1:
                            if control_assign==0:
                                self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                            elif control_assign==1:
                                self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                            elif control_assign==2:
                                self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)
                            if limited_capacity:
                                if capacity_assign == 0:
                                    self.ceo.distribute_capacity(even=True, weighted=False, included_last=True)
                                else:
                                    self.ceo.distribute_capacity(even=False, weighted=True, included_last=True)
                    else:
                        self.centralized_pivot(existing=None, under_control=1, independent=None)

                elif independent:

                    if not independent_keep:

                        if (step//change_interval) % 2==0:
                            self.decentralized_configure()

                            if ((step+change_interval//2)//change_interval) % 2==1:

                                if self.ceo.managers[-1].partner == -1:
                                    if limited_capacity:
                                        if capacity_assign==0:
                                            self.ceo.distribute_capacity(even=True, weighted=False, included_last=False)
                                        elif capacity_assign==1:
                                            self.ceo.distribute_capacity(even=False, weighted=True, included_last=False)
                                    self.ceo.hold_decision = list(self.ceo.managers[-1].decision)
                                    self.ceo.managers.pop(-1)
                                if independent_assign==0:
                                    self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                                elif independent_assign==1:
                                    self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                                elif independent_assign==2:
                                    self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)

                        else:
                            self.centralized_pivot(existing=None, under_control=None, independent=1)
                            if ((step+1)//change_interval) % 2 ==0:

                                if len(self.ceo.hold_decision)!=0:
                                    temp_manager = Manager(
                                        list(self.ceo.hold_decision), self.ceo.state, self.ceo.industry, self.ceo.managers[-1].capacity,
                                        -1 ,self.ceo, self.ceo.managers[0].constrain
                                    )
                                    self.ceo.managers.append(temp_manager)
                                    if limited_capacity:
                                        if capacity_assign==0:
                                            self.ceo.distribute_capacity(even=False, weighted=True, included_last=True)
                                        elif capacity_assign==1:
                                            self.ceo.distribute_capacity(even=True, weighted=False, included_last=True)

                                    self.ceo.hold_decision = []
                    else:
                        if (step//change_interval) % 2==0:
                            self.decentralized_configure()

                            if ((step+change_interval//2)//change_interval) % 2==1:

                                if self.ceo.managers[-1].partner == -1:

                                    if len(self.ceo.managers[-1].decision)>=self.ceo.managers[-1].constrain:
                                        self.ceo.managers[-1].partner = None
                                    else:
                                        if limited_capacity:
                                            if capacity_assign == 0:
                                                self.ceo.distribute_capacity(even=True, weighted=False,
                                                                             included_last=False)
                                            elif capacity_assign == 1:
                                                self.ceo.distribute_capacity(even=False, weighted=True,
                                                                             included_last=False)
                                        self.ceo.hold_decision = list(self.ceo.managers[-1].decision)
                                        self.ceo.managers.pop(-1)

                                if independent_assign == 0:
                                    self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                                elif independent_assign == 1:
                                    self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                                elif independent_assign == 2:
                                    self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)

                        else:
                            self.centralized_pivot(existing=None, under_control=None, independent=1)
                            if ((step+1)//change_interval) % 2 ==0:

                                if len(self.ceo.hold_decision)!=0:
                                    temp_manager = Manager(
                                        list(self.ceo.hold_decision), self.ceo.state, self.ceo.industry, self.ceo.managers[0].capacity,
                                        -1 ,self.ceo, self.ceo.managers[0].constrain
                                    )
                                    self.ceo.managers.append(temp_manager)
                                    if limited_capacity:
                                        if capacity_assign==0:
                                            self.ceo.distribute_capacity(even=True, weighted=False, included_last=True)
                                        elif capacity_assign==1:
                                            self.ceo.distribute_capacity(even=False, weighted=True, included_last=True)
                                    self.ceo.hold_decision = []
        else:
            if decentralized_pivot:
                # centralized configuration & decentralized pivot
                if (step//change_interval) % 2 == 0:
                    self.centralized_configuration(pivot_under_control=False)
                else:
                    self.decentralized_pivot()
            elif merge:
                if (step // change_interval) % 2 == 0:
                    self.centralized_configuration()
                else:
                    self.centralized_pivot(existing=merge_assign, under_control=None, independent=None)
            elif under_control:
                if (step // change_interval) % 2 == 0:
                    self.centralized_configuration(pivot_under_control=True)
                    if ((step + change_interval // 2) // change_interval) % 2 == 1:
                        if control_assign == 0:
                            self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                        elif control_assign == 1:
                            self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                        elif control_assign == 2:
                            self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)
                else:
                    self.centralized_pivot(existing=None, under_control=1, independent=None)
            elif independent:
                if not independent_keep:

                    if (step // change_interval) % 2 == 0:
                        self.centralized_configuration()

                        if ((step + change_interval // 2) // change_interval) % 2 == 1:

                            if self.ceo.managers[-1].partner == -1:
                                self.ceo.hold_decision = list(self.ceo.managers[-1].decision)
                                self.ceo.managers.pop(-1)
                            if independent_assign == 0:
                                self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                            elif independent_assign == 1:
                                self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                            elif independent_assign == 2:
                                self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)

                    else:
                        self.centralized_pivot(existing=None, under_control=None, independent=1)
                        if ((step + 1) // change_interval) % 2 == 0:

                            if len(self.ceo.hold_decision) != 0:
                                temp_manager = Manager(
                                    list(self.ceo.hold_decision), self.ceo.state, self.ceo.industry,
                                    self.ceo.managers[-1].capacity,
                                    -1, self.ceo, self.ceo.managers[0].constrain
                                )
                                self.ceo.managers.append(temp_manager)
                                self.ceo.hold_decision = []
                else:
                    if (step // change_interval) % 2 == 0:
                        self.centralized_configuration()

                        if ((step + change_interval // 2) // change_interval) % 2 == 1:

                            if self.ceo.managers[-1].partner == -1:

                                if len(self.ceo.managers[-1].decision) >= self.ceo.managers[-1].constrain:
                                    self.ceo.managers[-1].partner = None
                                else:
                                    self.ceo.hold_decision = list(self.ceo.managers[-1].decision)
                                    self.ceo.managers.pop(-1)

                            if independent_assign == 0:
                                self.ceo.reassign_controled_decision(random=True, even=False, relevant=False)
                            elif independent_assign == 1:
                                self.ceo.reassign_controled_decision(random=False, even=True, relevant=False)
                            elif independent_assign == 2:
                                self.ceo.reassign_controled_decision(random=False, even=False, relevant=True)

                    else:
                        self.centralized_pivot(existing=None, under_control=None, independent=1)
                        if ((step + 1) // change_interval) % 2 == 0:

                            if len(self.ceo.hold_decision) != 0:
                                temp_manager = Manager(
                                    list(self.ceo.hold_decision), self.ceo.state, self.ceo.industry,
                                    self.ceo.managers[0].capacity,
                                    -1, self.ceo, self.ceo.managers[0].constrain
                                )
                                self.ceo.managers.append(temp_manager)
                                self.ceo.hold_decision = []
        self.fitness_values = self.industry.query_fitness(self.ceo.state, self.ceo.decision)


def simulation(idx, return_dic, N, K, land_num, firm_num, period, altceo, altsub, constrain,
               decentralized_configuration=True, decentralized_pivot=True, merge=True,merge_assign=0, under_control=True,
               control_assign=0, independent=False, independent_keep=False, independent_assign=0, limited_capacity=False,
               capacity_assign=0):

    res_fitness = []

    for land in range(land_num):

        print(land)

        land_fitness = []

        np.random.seed(None)
        industry = Industry(N, K, None)
        industry.initialize(first_time=True, norm=True,)

        firms = []

        for cur in range(firm_num):
            firms.append(Firm(N, altceo, altsub, industry, constrain))

        for step in range(period):

            for firm in firms:
                firm.adaptation(
                    step, decentralized_configuration, decentralized_pivot, merge, merge_assign, under_control,
                    control_assign, independent, independent_keep, independent_assign, limited_capacity=False,
                    capacity_assign=0
                )

            if step % 5 ==0:
                land_fitness.append([firm.fitness_values for firm in firms])
        res_fitness.append(land_fitness)
        print(np.mean(np.array(res_fitness)[:, -1, :]))
    return_dic[idx] = np.array(res_fitness)







