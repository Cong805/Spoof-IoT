#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import numpy as np
import utils
import copy
import torch.nn.functional as F
np.random.seed(2)


class LGS_Attack(object):
    def __init__(self, model, pop_size=60, max_iters=20):
        self.model = model
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.query_count=0


    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3


    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))


    def predict(self, input_list):
        # print("input_list",input_list)
        input_list = utils.ori_preprocessing(" ".join(input_list))
        input = " ".join(input_list)
        # print(input)
        ori_pred, pred_head, pred_tail, rel_list = self.model(input)  # ori_batch_ent_shaking_outputs
        self.query_count += 1
        return torch.softmax(ori_pred.cpu().view(-1, ori_pred.size()[-1]), dim=1), rel_list, \
               torch.softmax(pred_head.cpu(), dim=3), \
               torch.softmax(pred_tail.cpu(), dim=3)


    def parse_rel_list(self, rel_list):
        device_type, brand, product = "", "", ""

        for rel in rel_list:
            if rel["predicate"] == "belong to":
                product = rel["subject"]
                device_type = rel["object"]
            elif rel["predicate"] == "produce":
                brand = rel["subject"]
                product = rel["object"]
            elif rel["predicate"] == "supply":
                brand = rel["subject"]
                device_type = rel["object"]
        return [device_type, brand, product]


    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new


    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new


    def predict_batch(self, sentences):
        # print(predict(sentences[0]))
        # return np.array([predict(s) for s in sentences])
        return torch.stack([self.predict(s)[0] for s in sentences],dim=0)


    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(np.array(x) != np.array(x_orig))) / float(x_len)
        return change_ratio


    def perturb_one(self, x_cur, x_cur_score, x_orig, neigbhours, w_select_probs, ori_label_mask,ori_tag):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neigbhours[rand_idx]
        rand_replace_id = np.random.choice(len(replace_list), 1)[0]
        while x_cur[rand_idx] == replace_list[rand_replace_id] and len(replace_list) > 1:
            rand_replace_id = np.random.choice(len(replace_list), 1)[0]

        x_new = x_cur.copy()
        x_new[rand_idx] = replace_list[rand_replace_id]

        x_pred, x_rel_list, _, _ = self.predict(x_new)
        x_tag = self.parse_rel_list(x_rel_list)
        # x_tag[0] != ori_tag[0] or x_tag[1] != ori_tag[1] or x_tag[2] != ori_tag[2]
        if x_tag[2] != ori_tag[2]:  #
            # if np.argmax(new_x_preds[best_id]) != target:
            return [1, x_new]
        if x_pred.shape[0] < ori_label_mask.shape[0]:
            x_score = 1 * ori_label_mask.sum() - x_pred[:, 1].masked_select(ori_label_mask[:x_pred.shape[0]]).sum()
        elif x_pred.shape[0] > ori_label_mask.shape[0]:
            if x_pred.shape[0] / ori_label_mask.shape[0] == 1.5:
                x_score = 1 * ori_label_mask.sum() - x_pred[:, 1].masked_select(
                    torch.cat((ori_label_mask, ori_label_mask[:int(ori_label_mask.shape[0] * 0.5)]), dim=0)).sum()
            else:
                x_score = 1 * ori_label_mask.sum() - x_pred[:, 1].masked_select(
                    torch.cat((ori_label_mask, ori_label_mask), dim=0)).sum()
        else:
            x_score = 1 * ori_label_mask.sum() - x_pred[:, 1].masked_select(ori_label_mask).sum()
        if x_score > x_cur_score:
            return [x_score, x_new]
        else:
            return [x_cur_score, x_cur]


    def generate_population(self,x_orig, ori_score, neigbhours_list, w_select_probs, ori_label_mask, pop_size,ori_tag):
        pop = []
        pop_scores = []
        for i in range(pop_size):
            tem = self.perturb_one(x_orig, ori_score, x_orig, neigbhours_list, w_select_probs, ori_label_mask,ori_tag)
            if tem is None:
                return None
            if tem[0] == 1:
                return [tem[1]]
            else:
                pop_scores.append(tem[0])
                pop.append(tem[1])
        return pop_scores, pop


    def attack(self,x_ori, perturbable_words_id, perturbable_words_substitution):

        ori_score,ori_rel,_,_ = self.predict(x_ori)  # ori_batch_ent_shaking_outputs
        self.query_count = 0
        # ori_score=ori_score.view(-1, ori_score.size()[-1])
        ori_label = torch.argmax(ori_score, dim=-1)
        ori_tag = self.parse_rel_list(ori_rel)

        ori_label_mask=ori_label.eq(1)
        # print("sum", ori_score[0])
        # print("sum",ori_label_mask.sum())
        ori_score=1*ori_label_mask.sum()-ori_score[:, 1].masked_select(ori_label_mask).sum()
        target = ori_label.masked_select(ori_label_mask)
        print("ori_tag:", ori_tag, "target:", target, 'ori:', ori_score, "perturbable num", len(perturbable_words_id))
        # print(label1.any()==False)
        # print(ori_score.shape,ori_label.shape)
        if not ori_rel:
            return None, self.query_count
        if not ori_label_mask.sum():
            return None,self.query_count

        neigbhours_list = []
        x_len = len(x_ori)
        for i in range(x_len):
            if i not in perturbable_words_id:
                neigbhours_list.append([])
                continue
            else:
                neigbhours_list.append(perturbable_words_substitution[i])
        neighbours_len = [len(x) for x in neigbhours_list]

        power = self.pop_size
        hotwordsid,hotwords = self.get_hot_word(x_ori, perturbable_words_id, power)
        # print("hotwordsid", x_len, hotwordsid)

        w_select_probs = []  # 重点词被选概率大
        w_select_probs1 = []  # 突变时可扰动词被选概率
        j = 0
        hotwordslen = len(hotwordsid)
        for i in range(x_len):
            if neighbours_len[i] == 0:
                w_select_probs.append(0)
                w_select_probs1.append(0)
            else:
                w_select_probs1.append(neighbours_len[i])
                if j < hotwordslen and i == hotwordsid[j]:
                    w_select_probs.append(neighbours_len[i] * 10)  # (power - j)
                    j += 1
                    continue
                if x_ori[i] in hotwords:
                    w_select_probs.append(neighbours_len[i] * 10)  # (20 - hotwords.index(x_ori[i]))
                    continue
                w_select_probs.append(neighbours_len[i])
        w_select_probs = w_select_probs / np.sum(w_select_probs)
        w_select_probs1 = w_select_probs1 / np.sum(w_select_probs1)
        # print("w_select_probs",w_select_probs)

        tem = self.generate_population(x_ori, ori_score, perturbable_words_substitution, w_select_probs, ori_label_mask, self.pop_size,ori_tag)
        if tem is None:
            return None,self.query_count
        if len(tem) == 1:
            return tem[0],self.query_count
        pop_scores, pop = tem
        part_elites = copy.deepcopy(pop)
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[-1]
        all_elite = pop[top_attack]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):

            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)

            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                            self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                              all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)
                if np.random.uniform() < P1:  # PI
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:  # PG
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            pop_scores = []
            pop_scores_all = []
            pop_tags_all=[]
            for a in pop:
                pt,pt_tag,_,_ = self.predict(a)#[:, 1].masked_select(ori_label_mask).sum()

                if pt.shape[0] < ori_label_mask.shape[0]:
                    pt_score = 1 * ori_label_mask.sum() - pt[:, 1].masked_select(
                        ori_label_mask[:pt.shape[0]]).sum()
                elif pt.shape[0] > ori_label_mask.shape[0]:
                    if pt.shape[0] / ori_label_mask.shape[0] == 1.5:
                        pt_score = 1 * ori_label_mask.sum() - pt[:, 1].masked_select(
                            torch.cat((ori_label_mask, ori_label_mask[:int(ori_label_mask.shape[0] * 0.5)]),
                                      dim=0)).sum()
                    else:
                        pt_score = 1 * ori_label_mask.sum() - pt[:, 1].masked_select(
                            torch.cat((ori_label_mask, ori_label_mask), dim=0)).sum()
                else:
                    pt_score = 1 * ori_label_mask.sum() - pt[:, 1].masked_select(ori_label_mask).sum()
                pop_scores.append(pt_score)
                pop_scores_all.append(pt)
                pop_tags_all.append(self.parse_rel_list(pt_tag))
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]

            print('\t\t', i, ' -- ', pop_scores[top_attack],ori_label_mask.sum())

            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                # if torch.argmax(pt, dim=-1).masked_select(ori_label_mask).sum()!=ori_label_mask.sum():#!=target:
                # pop_tags_all[pt_id][0]!=ori_tag[0] or pop_tags_all[pt_id][1]!=ori_tag[1] or pop_tags_all[pt_id][2]!=ori_tag[2]
                if pop_tags_all[pt_id][2]!=ori_tag[2]:
                    return pop[pt_id],self.query_count
                if pop_scores[pt_id]==ori_label_mask.sum() and pt.shape[0]==ori_label_mask.shape[0]:
                    ori_label_mask=torch.logical_or(torch.argmax(pt, dim=-1).eq(1),ori_label_mask)

            new_pop = []
            new_pop_scores = []
            for id in range(len(pop)):
                x = pop[id]
                change_ratio = self.count_change_ratio(x, x_ori, x_len)
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    tem = self.perturb_one(x,pop_scores[id], x_ori, neigbhours_list, w_select_probs1,ori_label_mask,ori_tag)
                    if tem is None:
                        return None,self.query_count
                    if tem[0] == 1:
                        return tem[1],self.query_count
                    else:
                        new_pop_scores.append(tem[0])
                        new_pop.append(tem[1])
                else:
                    new_pop_scores.append(pop_scores[id])
                    new_pop.append(x)
            pop = new_pop
            pop_scores = new_pop_scores
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)
        return None, self.query_count

    def get_hot_word(self, x_ori, perturbable_words_id, power):
        batch_ent_shaking_outputs, _,head_out,tial_out = self.predict(x_ori)
        batch_ent_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim=-1)
        head_tag = torch.argmax(head_out, dim=-1)
        tail_tag = torch.argmax(tial_out, dim=-1)

        losses = self.calculate_conf(x_ori, perturbable_words_id, batch_ent_shaking_tag,head_tag,tail_tag)
        # print(losses,losses.shape)
        sorted, indices = torch.sort(losses, descending=True)
        hotwordsid=[]

        for i in indices:
            hotwordsid.append(perturbable_words_id[i])
            if len(hotwordsid) >= power:
                break
        # wordsid = list(set(wordsid))
        hotwords = [x_ori[i] for i in hotwordsid]
        print(hotwordsid,hotwords)
        return hotwordsid,hotwords  # wordsid

    def calculate_conf(self, x_ori, perturbable_words_id, pred,pred_head,pred_tail):
        l = len(perturbable_words_id)
        losses = torch.zeros(l)
        for i in range(l):
            tempinputs = x_ori.copy()
            tempinputs[perturbable_words_id[i]] = "<UNK>"
            with torch.no_grad():
                input_unk = tempinputs
                ent_shaking_outputs, _,head_rel_shaking_outputs,tail_rel_shaking_outputs = self.predict(input_unk)
                # print("11",ent_shaking_outputs[1].sum(),ent_shaking_outputs.shape)
                # ent_shaking_outputs = ent_shaking_outputs.view(-1, ent_shaking_outputs.size()[-1])
                # print("tmp", pred_head[:, 0, :].reshape(-1,1).view(-1).shape, pred.shape)
                pred_head_r0 = pred_head[:, 1, :].reshape(-1,1).view(-1)
                # pred_head_r2 = pred_head[:, 2, :].reshape(-1,1).view(-1)
                pred_head_r1 = pred_head[:, 0, :].reshape(-1, 1).view(-1)
                head_outputs_r0 = head_rel_shaking_outputs[:, 1, :, :]
                head_outputs_r0 = head_outputs_r0.reshape(head_outputs_r0.shape[0]*head_outputs_r0.shape[1],head_outputs_r0.shape[2])
                # head_outputs_r2 = head_rel_shaking_outputs[:, 2, :, :]
                # head_outputs_r2 = head_outputs_r2.reshape(head_outputs_r2.shape[0]*head_outputs_r2.shape[1],head_outputs_r2.shape[2])
                head_outputs_r1 = head_rel_shaking_outputs[:, 0, :, :]
                head_outputs_r1 = head_outputs_r1.reshape(head_outputs_r1.shape[0] * head_outputs_r1.shape[1], head_outputs_r1.shape[2])
                # # print("pred_head_r0", pred_head_r0.shape, pred_head_r0)
                #
                # # print("head_rel_shaking_outputs[:, 0, :, :]",head_rel_shaking_outputs[:, 0, :, :].shape)
                pred_tail_r0 = pred_tail[:, 1, :].reshape(-1,1).view(-1)
                # pred_tail_r2 = pred_tail[:, 2, :].reshape(-1,1).view(-1)
                pred_tail_r1 = pred_tail[:, 0, :].reshape(-1, 1).view(-1)
                tail_outputs_r0 = tail_rel_shaking_outputs[:, 1, :, :]
                tail_outputs_r0 = tail_outputs_r0.reshape(tail_outputs_r0.shape[0]*tail_outputs_r0.shape[1],tail_outputs_r0.shape[2])
                # tail_outputs_r2 = tail_rel_shaking_outputs[:, 2, :, :]
                # tail_outputs_r2 = tail_outputs_r2.reshape(tail_outputs_r2.shape[0]*tail_outputs_r2.shape[1],tail_outputs_r2.shape[2])
                tail_outputs_r1 = tail_rel_shaking_outputs[:, 0, :, :]
                tail_outputs_r1 = tail_outputs_r1.reshape(tail_outputs_r1.shape[0] * tail_outputs_r1.shape[1],tail_outputs_r1.shape[2])

                # print("tmp", batch_ent_shaking_outputs.shape, pred.shape)

            if pred.shape[0] < ent_shaking_outputs.shape[0]:
                # losses[i] = F.nll_loss(ent_shaking_outputs, torch.cat((pred, pred), dim=0), reduction='none').sum()
                losses[i] = F.nll_loss(ent_shaking_outputs, torch.cat((pred, pred), dim=0), reduction='none').sum()+\
                            F.nll_loss(head_outputs_r0, torch.cat((pred_head_r0, pred_head_r0), dim=0), reduction='none').sum()+\
                            F.nll_loss(tail_outputs_r0, torch.cat((pred_tail_r0, pred_tail_r0), dim=0),reduction='none').sum() + \
                            F.nll_loss(head_outputs_r1, torch.cat((pred_head_r1, pred_head_r1), dim=0),reduction='none').sum() + \
                            F.nll_loss(tail_outputs_r1, torch.cat((pred_tail_r1, pred_tail_r1), dim=0),reduction='none').sum()
                #             F.nll_loss(head_outputs_r2, torch.cat((pred_head_r2, pred_head_r2), dim=0), reduction='none').sum()+ \
                #             F.nll_loss(tail_outputs_r2, torch.cat((pred_tail_r2, pred_tail_r2), dim=0),reduction='none').sum()+ \
            elif pred.shape[0] > ent_shaking_outputs.shape[0]:
                # losses[i] = F.nll_loss(ent_shaking_outputs, pred[:ent_shaking_outputs.shape[0]], reduction='none').sum()
                losses[i] = F.nll_loss(ent_shaking_outputs, pred[:ent_shaking_outputs.shape[0]], reduction='none').sum()+ \
                            F.nll_loss(head_outputs_r0, pred_head_r0[:head_outputs_r0.shape[0]], reduction='none').sum()+ \
                            F.nll_loss(tail_outputs_r0, pred_tail_r0[:tail_outputs_r0.shape[0]], reduction='none').sum()+ \
                            F.nll_loss(head_outputs_r1, pred_head_r1[:head_outputs_r1.shape[0]],reduction='none').sum() + \
                            F.nll_loss(tail_outputs_r1, pred_tail_r1[:tail_outputs_r1.shape[0]],reduction='none').sum()
                            # F.nll_loss(tail_outputs_r2, pred_tail_r2[:tail_outputs_r2.shape[0]], reduction='none').sum()+ \
                #             F.nll_loss(head_outputs_r2, pred_head_r2[:head_outputs_r2.shape[0]], reduction='none').sum()+ \
            else:
                # losses[i] = F.nll_loss(ent_shaking_outputs, pred, reduction='none').sum()
                losses[i] = F.nll_loss(ent_shaking_outputs, pred, reduction='none').sum()+ \
                            F.nll_loss(head_outputs_r0, pred_head_r0, reduction='none').sum()+ \
                            F.nll_loss(tail_outputs_r0, pred_tail_r0, reduction='none').sum()+ \
                            F.nll_loss(head_outputs_r1, pred_head_r1, reduction='none').sum()+ \
                            F.nll_loss(tail_outputs_r1, pred_tail_r1, reduction='none').sum()
        #             F.nll_loss(head_outputs_r2, pred_head_r2, reduction='none').sum()+ \
                #             F.nll_loss(tail_outputs_r2, pred_tail_r2, reduction='none').sum()+ \
        return losses


if __name__ == '__main__':
    pass
