import json
from tqdm import tqdm
import re
import sys
import re
import os

class VQAEval:
    def __init__(self, vqa_eval_file=None, n=2):
        # vqa_eval_file is a json file format like 
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa_eval_res = json.load(open(vqa_eval_file, "r"))
        # self.vqa = vqa
        # self.vqaRes = vqaRes
        # if vqa is not None:
        #     self.params = {"question_id": vqa.getQuesIds()}
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
        
    def evaluate_wzq_balanced(self, quesIds=None):
        # may error
            quesIds = [res_json['question_id'] for res_json in self.vqa_eval_res]
            # if quesIds == None:
            #     quesIds = [quesId for quesId in self.params["question_id"]]
            gts = {}
            res = {}
            # from here to get gts and res. In ours format, they should be same.
            # for quesId in quesIds:
            #     gts[quesId] = self.vqa.qa[quesId]
            #     res[quesId] = self.vqaRes.qa[quesId]
            for res_json in self.vqa_eval_res:
                gts[res_json['question_id']] = res_json
                res[res_json['question_id']] = res_json
            # =================================================
            # Compute accuracy
            # =================================================
            accQA = []
            accQuesType = {}
            accAnsType = {}
            print("computing accuracy")
            step = 0
            for quesId in quesIds:
                resAns = res[quesId]["balanced_answer"]
                resAns = resAns.replace("\n", " ")
                resAns = resAns.replace("\t", " ")
                resAns = resAns.strip()
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)
                gtAcc = []
                gtAnswers = [ans for ans in gts[quesId]["answers"]]
                if len(set(gtAnswers)) > 1:
                    for ansDic in gts[quesId]["answers"]:
                        ansDic = self.processPunctuation(ansDic)
                # print('ansDic', ansDic)
                # for gtAnsDatum in gts[quesId]["answers"]:
                matchingAns = [item for item in gtAnswers if item == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
                quesType = gts[quesId]["question_type"]
                ansType = gts[quesId]["answer_type"]
                avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
                accQA.append(avgGTAcc)
                if quesType not in accQuesType:
                    accQuesType[quesType] = []
                accQuesType[quesType].append(avgGTAcc)
                if ansType not in accAnsType:
                    accAnsType[ansType] = []
                accAnsType[ansType].append(avgGTAcc)
                self.setEvalQA(quesId, avgGTAcc)
                self.setEvalQuesType(quesId, quesType, avgGTAcc)
                self.setEvalAnsType(quesId, ansType, avgGTAcc)
                if step % 100 == 0:
                    self.updateProgress(step / float(len(quesIds)))
                step = step + 1
                # assert False
            self.setAccuracy(accQA, accQuesType, accAnsType)
            print("Done computing accuracy")
            print("acc:", self.accuracy["overall"])

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        print('over all len:', len(accQA))
        self.accuracy["perQuestionType"] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy["perAnswerType"] = {
            ansType: round(
                100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n
            )
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[ quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), int(progress * 100), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def get_vqa_score(gt_ans, pred_ans):
    pred_ans = pred_ans.replace("\n", " ")
    pred_ans = pred_ans.replace("\t", " ")
    pred_ans = pred_ans.strip()
    pred_ans = processPunctuation(pred_ans)
    pred_ans = processDigitArticle(pred_ans)
    if len(set(gt_ans)) > 1:
        for ansDic in gt_ans:
            ansDic = processPunctuation(ansDic)
    matchingAns = [item for item in gt_ans if item == pred_ans]
    score = min(1, float(len(matchingAns)) / 3)
    return score


def filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count):
    # assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[0][0] - list(x[1]['statements_prob_norm'].values())[0][1]) > assist_query_difference, assist_info.items()))
    # assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[1][0] - list(x[1]['statements_prob_norm'].values())[1][1]) > assist_query_difference, assist_info.items()))
    # assist_info = dict(filter(lambda x: (list(x[1]['statements_prob_norm'].values())[0][0] >= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] >= 0.5) or (list(x[1]['statements_prob_norm'].values())[0][0] <= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] <= 0.5), assist_info.items()))
    new_assist_info = {}
    for query, query_info in assist_info.items():
        try:
            statements_prob_norm_values = list(query_info['statements_prob_norm'].values())
            # print(statements_prob_norm_values)
            if abs(statements_prob_norm_values[0][0] - statements_prob_norm_values[0][1]) > assist_query_difference and \
                abs(statements_prob_norm_values[1][0] - statements_prob_norm_values[1][1]) > assist_query_difference and \
                ((statements_prob_norm_values[0][0] >= 0.5 and statements_prob_norm_values[1][1] >= 0.5) or \
                (statements_prob_norm_values[0][0] <= 0.5 and statements_prob_norm_values[1][1] <= 0.5)):
                new_assist_info[query] = query_info
        
        except Exception as e:
            error_count = error_count + 1
            continue
    return new_assist_info, error_count


def filter_by_assist_candidates_difference(assist_info, assist_candidates_difference):
    # assist_info = dict(filter(lambda x: abs(list(x[1]['assist_candidates'].values())[0] - list(x[1]['assist_candidates'].values())[1]) > assist_candidates_difference, assist_info.items()))
    new_assist_info = {}
    for query, query_info in assist_info.items():
        assist_candidates_values = list(query_info['assist_candidates'].values())
        if abs(assist_candidates_values[0] - assist_candidates_values[1]) > assist_candidates_difference:
            new_assist_info[query] = query_info
    return new_assist_info


def get_balanced_llm_prob(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            res['balanced_candidates'] = 'no need to balance because the top2 candidates is equal '
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        # 根据top2间的candidates差距判断是否需要balance
        difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
        if difference > candidates_difference:
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
            continue
        
        assist_info = res['assist_info']
        # assist_candidates概率差距要大于assist_candidates_difference
        assist_info = filter_by_assist_candidates_difference(assist_info, assist_candidates_difference)
        # 每对statements_prob_norm的差距要大于assist_query_difference
        assist_info, error_count = filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count)

        assist_queries = list(assist_info.keys())
        if len(assist_queries) == 0:
            res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
            res['balanced_answer'] = res['pred_ans']
            continue
        
        max_assist_candidates_difference = 0.0
        max_score = 0.0
        max_assist_candidates_difference_query = ''
        for assist_query in assist_queries:
            assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
            try:
                # 选择llm_prob的概率差距最大的作为最终的assist_query
                if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query
            except Exception as e:
                error_count = error_count + 1
                continue

        # begin to balance 
        try:
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1

        except Exception as e:
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue
            
    print('error count:', error_count)
    return query_reses, balance_count
    pass


def get_balanced_candidates(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            res['balanced_candidates'] = 'no need to balance because the top2 candidates is equal '
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        # 根据top2间的candidates差距判断是否需要balance
        difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
        if difference > candidates_difference:
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
            continue
        
        assist_info = res['assist_info']
        # assist_candidates概率差距要大于assist_candidates_difference
        assist_info = filter_by_assist_candidates_difference(assist_info, assist_candidates_difference)
        # 每对statements_prob_norm的差距要大于assist_query_difference
        assist_info, error_count = filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count)
        
        assist_queries = list(assist_info.keys())
        if len(assist_queries) == 0:
            res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
            res['balanced_answer'] = res['pred_ans']
            continue
        
        max_assist_candidates_difference = 0.0
        max_score = 0.0
        max_assist_candidates_difference_query = ''
        for assist_query in assist_queries:
            assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())

            # 选择assist_candidates的概率差距最大的作为最终的assist_query
            try:
                if abs(assist_info[assist_query]['assist_candidates'][assist_candidates[0]] - assist_info[assist_query]['assist_candidates'][assist_candidates[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['assist_candidates'][assist_candidates[0]] - assist_info[assist_query]['assist_candidates'][assist_candidates[1]])
                    max_assist_candidates_difference_query = assist_query
            except Exception as e:
                error_count = error_count + 1
                continue

            # begin to balance 
        try:
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
            
        except Exception as e:
            # print('error', e)
            # print(res['image_id'])
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue

    print('error count:', error_count)
    return query_reses, balance_count
    pass


def get_balanced_score(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        
        assist_info = res['assist_info']
        assist_queries = list(assist_info.keys())
        
        max_assist_candidates_difference = 0.0
        max_score = 0.0
        max_assist_candidates_difference_query = ''
        gt_ans = res["answers"]
        for assist_query in assist_queries:
            assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())

            # 选llm_prob在vqa单条评价上分数高的为最终的assist_query
            try:
                score = get_vqa_score(gt_ans, max(assist_info[assist_query]['llm_prob'], key=assist_info[assist_query]['llm_prob'].get))
                if score >= max_score:
                    max_score = score
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query
                # 选llm_prob的概率差距大的
                elif score == max_score:
                    if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                        max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                        max_assist_candidates_difference_query = assist_query

            except Exception as e:
                error_count = error_count + 1
                continue

        # begin to balance 
        try:
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
        except Exception as e:
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue
    print('error count:', error_count)
    return query_reses, balance_count
    pass


def get_balanced_llm_prob_discrete(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        try:
            candidates_dict = list(res['candidates_dict'].keys())
            # 根据top2间的candidates差距判断是否需要balance
            difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
            if difference > candidates_difference:
                res['balanced_answer'] = res['pred_ans']
                res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
                continue
            
            assist_info = res['assist_info']
            # assist_candidates概率差距要大于assist_candidates_difference
            assist_info = dict(filter(lambda x: abs(list(x[1]['assist_candidates'].values())[0] - list(x[1]['assist_candidates'].values())[1]) > assist_candidates_difference, assist_info.items()))
            # 每对statements_prob_norm的差距要大于assist_query_difference
            assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[0][0] - list(x[1]['statements_prob_norm'].values())[0][1]) > assist_query_difference, assist_info.items()))
            assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[1][0] - list(x[1]['statements_prob_norm'].values())[1][1]) > assist_query_difference, assist_info.items()))
            assist_info = dict(filter(lambda x: (list(x[1]['statements_prob_norm'].values())[0][0] >= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] >= 0.5) or (list(x[1]['statements_prob_norm'].values())[0][0] <= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] <= 0.5), assist_info.items()))
            assist_queries = list(assist_info.keys())
            if len(assist_queries) == 0:
                res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
                res['balanced_answer'] = res['pred_ans']
                continue
            
            max_assist_candidates_difference = 0.0
            max_score = 0.0
            max_assist_candidates_difference_query = ''
            for assist_query in assist_queries:
                assist_info[assist_query]['origin_llm_prob'] = assist_info[assist_query]['llm_prob']
                assist_info[assist_query]['llm_prob'] = {}
                assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                assist_candidates_0_pro = 1 if assist_info[assist_query]['assist_candidates'][assist_candidates[0]] > 0.5 else 0
                assist_candidates_1_pro = 1 if assist_candidates_0_pro==0 else 0
                candidates_1_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][0] * assist_candidates_0_pro \
                                        + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][0] * assist_candidates_1_pro
                candidates_2_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][1] * assist_candidates_0_pro \
                                        + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][1] * assist_candidates_1_pro
                assist_info[assist_query]['llm_prob'][candidates_dict[0]] = candidates_1_probability
                assist_info[assist_query]['llm_prob'][candidates_dict[1]] = candidates_2_probability

                # 选择llm_prob的概率差距最大的作为最终的assist_query
                if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query

            # use the first assist query
            # max_assist_candidates_difference_query = assist_queries[0]
            # begin to balance 
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
            pass
        except Exception as e:
            # print('error', e)
            # print(res['image_id'])
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue
    print('error count:', error_count)
    return query_reses, balance_count
    pass


def is_consistent_trend(assist_info):
    origin_len = len(assist_info)
    if len(dict(filter(lambda x: (list(x[1]['llm_prob'].values())[0] > 0.5), assist_info.items()))) == origin_len:
        return True
    if len(dict(filter(lambda x: (list(x[1]['llm_prob'].values())[0] < 0.5), assist_info.items()))) == origin_len:
        return True
    return False


def get_balanced_llm_prob_trend(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        try:
            candidates_dict = list(res['candidates_dict'].keys())
            # 根据top2间的candidates差距判断是否需要balance
            difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
            if difference > candidates_difference:
                res['balanced_answer'] = res['pred_ans']
                res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
                continue
            
            assist_info = res['assist_info']

            if not is_consistent_trend(assist_info):
                # assist_candidates概率差距要大于assist_candidates_difference
                assist_info = dict(filter(lambda x: abs(list(x[1]['assist_candidates'].values())[0] - list(x[1]['assist_candidates'].values())[1]) > assist_candidates_difference, assist_info.items()))
                # 每对statements_prob_norm的差距要大于assist_query_difference
                assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[0][0] - list(x[1]['statements_prob_norm'].values())[0][1]) > assist_query_difference, assist_info.items()))
                assist_info = dict(filter(lambda x: abs(list(x[1]['statements_prob_norm'].values())[1][0] - list(x[1]['statements_prob_norm'].values())[1][1]) > assist_query_difference, assist_info.items()))
                assist_info = dict(filter(lambda x: (list(x[1]['statements_prob_norm'].values())[0][0] >= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] >= 0.5) or (list(x[1]['statements_prob_norm'].values())[0][0] <= 0.5 and list(x[1]['statements_prob_norm'].values())[1][1] <= 0.5), assist_info.items()))
                
            assist_queries = list(assist_info.keys())

            if len(assist_queries) == 0:
                res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
                res['balanced_answer'] = res['pred_ans']
                continue
            
            max_assist_candidates_difference = 0.0
            max_score = 0.0
            max_assist_candidates_difference_query = ''
            for assist_query in assist_queries:
                assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())

                # 选择llm_prob的概率差距最大的作为最终的assist_query
                if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query

            # use the first assist query
            # max_assist_candidates_difference_query = assist_queries[0]
            # begin to balance 
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
            pass
        except Exception as e:
            # print('error', e)
            # print(res['image_id'])
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue
    print('error count:', error_count)
    return query_reses, balance_count
    pass


def get_balanced_llm_prob_yes(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            res['balanced_candidates'] = 'no need to balance because the top2 candidates is equal '
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        # 根据top2间的candidates差距判断是否需要balance
        difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
        if difference > candidates_difference:
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
            continue
        
        assist_info = res['assist_info']
        assist_info_len = len(assist_info)
        assist_info_value = list(assist_info.values())
        assist_info_key = list(assist_info.keys())
        new_assist_info = {}

        have_yes = False
        for idx in range(assist_info_len):
            if max(assist_info_value[idx]['assist_candidates'], key=assist_info_value[idx]['assist_candidates'].get) == "yes":
                have_yes = True
                new_assist_info[assist_info_key[idx]] = assist_info_value[idx]

        if not have_yes:
            # assist_candidates概率差距要大于assist_candidates_difference
            assist_info = filter_by_assist_candidates_difference(assist_info, assist_candidates_difference)
            # 每对statements_prob_norm的差距要大于assist_query_difference
            assist_info, error_count = filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count)
        else:    
            assist_info = new_assist_info
        
        assist_queries = list(assist_info.keys())

        if len(assist_queries) == 0:
            res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
            res['balanced_answer'] = res['pred_ans']
            continue
        
        max_assist_candidates_difference = 0.0
        max_score = 0.0
        max_assist_candidates_difference_query = ''
        for assist_query in assist_queries:
            assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
            try:
            # 选择llm_prob的概率差距最大的作为最终的assist_query
                if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query
            except Exception as e:
                error_count = error_count + 1
                continue
        # begin to balance 
        try:
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
            
        except Exception as e:
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue
            
    print('error count:', error_count)
    return query_reses, balance_count


def get_balanced_llm_prob_yes2(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            res['balanced_candidates'] = 'no need to balance because the top2 candidates is equal '
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        # 根据top2间的candidates差距判断是否需要balance
        difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
        if difference > candidates_difference:
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
            continue

        assist_info = res['assist_info']

        # assist_candidates概率差距要大于assist_candidates_difference
        assist_info = filter_by_assist_candidates_difference(assist_info, assist_candidates_difference)
        # 每对statements_prob_norm的差距要大于assist_query_difference
        assist_info, error_count = filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count)

        assist_info_len = len(assist_info)
        assist_info_value = list(assist_info.values())
        assist_info_key = list(assist_info.keys())
        new_assist_info = {}

        have_yes = False
        for idx in range(assist_info_len):
            if max(assist_info_value[idx]['assist_candidates'], key=assist_info_value[idx]['assist_candidates'].get) == "yes":
                have_yes = True
                new_assist_info[assist_info_key[idx]] = assist_info_value[idx]

        if have_yes:
            assist_info = new_assist_info

        assist_queries = list(assist_info.keys())

        if len(assist_queries) == 0:
            res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
            res['balanced_answer'] = res['pred_ans']
            continue
        
        max_assist_candidates_difference = 0.0
        max_score = 0.0
        max_assist_candidates_difference_query = ''
        for assist_query in assist_queries:
            assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
            try: 
                # 选择llm_prob的概率差距最大的作为最终的assist_query
                if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                    max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                    max_assist_candidates_difference_query = assist_query
            except Exception as e:
                error_count = error_count + 1
                continue

        # begin to balance
        try: 
            res['balanced_candidates'] = {}
            res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
            res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
            res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
            res['chosen_assist_query'] = max_assist_candidates_difference_query
            balance_count += 1
        except Exception as e:
            error_count = error_count + 1
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'something wrong, fail to balance'
            continue

    print('error count:', error_count)
    return query_reses, balance_count


def get_balanced_llm_prob_yes3(query_reses, vqa_confidence, candidates_difference, assist_query_difference, assist_candidates_difference):
    balance_count = 0
    error_count = 0
    llm_confidence = 1 - vqa_confidence
    for res in query_reses:
        if res['equal_answer'] == True:
            res['balanced_candidates'] = 'no need to balance because the top2 candidates is equal '
            continue
        candidates_dict = list(res['candidates_dict'].keys())
        # 根据top2间的candidates差距判断是否需要balance
        difference = abs(res['candidates_dict'][candidates_dict[0]] - res['candidates_dict'][candidates_dict[1]])
        if difference > candidates_difference:
            res['balanced_answer'] = res['pred_ans']
            res['balanced_candidates'] = 'no need to balance because the difference between top2 candidates greater than ' + str(candidates_difference)
            continue
        
        assist_info = res['assist_info']

        # assist_candidates概率差距要大于assist_candidates_difference
        assist_info = filter_by_assist_candidates_difference(assist_info, assist_candidates_difference)
        # 每对statements_prob_norm的差距要大于assist_query_difference
        assist_info, error_count = filter_query_by_statements_prob_norm(assist_info, assist_query_difference, error_count)

        assist_info_len = len(assist_info)
        assist_info_value = list(assist_info.values())
        assist_info_key = list(assist_info.keys())
        new_assist_info = {}

        have_yes = False
        for idx in range(assist_info_len):
            if max(assist_info_value[idx]['assist_candidates'], key=assist_info_value[idx]['assist_candidates'].get) == "yes":
                have_yes = True
                new_assist_info[assist_info_key[idx]] = assist_info_value[idx]

        if have_yes:
            assist_info = new_assist_info
            assist_queries = list(assist_info.keys())

            if len(assist_queries) == 0:
                res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
                res['balanced_answer'] = res['pred_ans']
                continue
            
            total_llm_prob_candidate_0 = 0.0
            total_llm_prob_candidate_1 = 0.0
            for assist_query in assist_queries:
                try:
                    assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                    total_llm_prob_candidate_0 += assist_info[assist_query]['llm_prob'][candidates_dict[0]]
                    total_llm_prob_candidate_1 += assist_info[assist_query]['llm_prob'][candidates_dict[1]]
                except Exception as e:
                    error_count = error_count + 1
                    continue

            # begin to balance 
            try:
                avg_llm_prob_candidate_0 = total_llm_prob_candidate_0 / len(assist_queries)
                avg_llm_prob_candidate_1 = total_llm_prob_candidate_1 / len(assist_queries)
                res['balanced_candidates'] = {}
                res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + avg_llm_prob_candidate_0 * llm_confidence
                res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + avg_llm_prob_candidate_1 * llm_confidence
                res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
                res['chosen_assist_query'] = assist_queries
                balance_count += 1
            except Exception as e:
                error_count = error_count + 1
                res['balanced_answer'] = res['pred_ans']
                res['balanced_candidates'] = 'something wrong, fail to balance'
                continue

        else:
            assist_queries = list(assist_info.keys())

            if len(assist_queries) == 0:
                res['balanced_candidates'] = 'choose not to balance because the assist_info cannot help'
                res['balanced_answer'] = res['pred_ans']
                continue
            
            max_assist_candidates_difference = 0.0
            max_score = 0.0
            max_assist_candidates_difference_query = ''
            for assist_query in assist_queries:
                try:
                    assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                    # 选择llm_prob的概率差距最大的作为最终的assist_query
                    if abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]]) > max_assist_candidates_difference:
                        max_assist_candidates_difference = abs(assist_info[assist_query]['llm_prob'][candidates_dict[0]] - assist_info[assist_query]['llm_prob'][candidates_dict[1]])
                        max_assist_candidates_difference_query = assist_query

                except Exception as e:
                    error_count = error_count + 1
                    continue

            # begin to balance 
            try:
                res['balanced_candidates'] = {}
                res['balanced_candidates'][candidates_dict[0]] = res['candidates_dict'][candidates_dict[0]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[0]] * llm_confidence
                res['balanced_candidates'][candidates_dict[1]] = res['candidates_dict'][candidates_dict[1]] * vqa_confidence + assist_info[max_assist_candidates_difference_query]['llm_prob'][candidates_dict[1]] * llm_confidence
                res['balanced_answer'] = max(res['balanced_candidates'], key=res['balanced_candidates'].get)
                res['chosen_assist_query'] = max_assist_candidates_difference_query
                balance_count += 1
            except Exception as e:
                error_count = error_count + 1
                res['balanced_answer'] = res['pred_ans']
                res['balanced_candidates'] = 'something wrong, fail to balance'
                continue

    print('error count:', error_count)
    return query_reses, balance_count

