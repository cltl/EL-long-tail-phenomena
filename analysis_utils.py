from scipy.stats import pearsonr, spearmanr
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from scipy.stats import linregress
import math

################## RETRIEVAL AND ANALYSIS #######################################

def calculate_slope(cnt):
        """
        Calculate the slope of the frequency curve. Ideal Zipfian distribution has slope of -1.0.
        """
        y = OrderedDict(cnt.most_common())
        v=np.log(list(y.values()))
        k=np.log(np.arange(1,len(v)+1,1))
        return linregress(k,v)

def get_mention_counts(articles, skip_nils=True):
        """
        Obtain frequency counters of both instances and forms.
        """
        gold_forms=[]
        gold_links=[]
        for example_article in articles:
                for entity in example_article.entity_mentions:
                        mention=entity.mention
                        meaning=entity.gold_link
                        if not skip_nils or meaning!='--NME--':
                                gold_forms.append(mention)
                                gold_links.append(meaning)
        cnt_instances=Counter(gold_links)
        cnt_forms=Counter(gold_forms)
        return cnt_instances, cnt_forms

def get_form_distribution(articles, the_form):
        """
        Get frequency-ordered instance distribution for a specific form.
        """
        instances = get_inst_with_counts(articles, the_form)
        return sorted(instances.items(), key=lambda x: x[1], reverse=True)

def get_form_counts(articles, the_form):
        """
        Get a list of instance frequency counts for a form.
        """
        instances = get_inst_with_counts(articles, the_form)
        return instances.values()

def get_inst_with_counts(articles, the_form): 
        """
        Get all instances with their corresponding occurence counts, for a given form.
        """
        instances = defaultdict(int) 
        for article in articles: 
                for mention in article.entity_mentions: 
                        form=mention.mention
                        meaning=mention.gold_link
                        if form==the_form and meaning!='--NME--':
                                instances[meaning]+=1
        return instances

def get_instance_distribution(articles, instance):
        """
        Get frequency-ordered form distribution for a specific instance.
        """
        references = defaultdict(int)
        for article in articles:
                for mention in article.entity_mentions:
                        form=mention.mention
                        meaning=mention.gold_link
                        if meaning==instance:
                                references[form]+=1
        return sorted(references.items(), key=lambda x: x[1], reverse=True)

def get_pageranks(articles, skip_zeros=False, ambiguous_only=False, ambiguous_forms=set()):
        """
        Obtain PageRank values and store in different dictionaries.
        """
        pageranks = {}
        pagerank_frequency=defaultdict(int)

        pr_uniq_sets=defaultdict(set)
        for article in articles:
                for mention in article.entity_mentions:
                        if ambiguous_only and mention.mention not in ambiguous_forms:
                            continue
                        h=int(mention.gold_pr/1)
                        if not skip_zeros or h!=0:
                                pagerank_frequency[h]+=1
                                pr_uniq_sets[h].add(mention.gold_link)
                        pageranks[mention.gold_link]=h
        pr_uniq=defaultdict(int)
        for k,v in pr_uniq_sets.items():
                pr_uniq[k]=len(v)
        return pagerank_frequency, pr_uniq, pageranks

def get_interpretations_and_references(articles, skip_nils=True):
        """
        Get sets of interpretations for all forms, and sets of referring expressions for all instances.
        """
        interpretations=defaultdict(set)
        references = defaultdict(set)
        for article in articles:
                for mention in article.entity_mentions:
                        form=mention.mention
                        meaning=mention.gold_link
                        if not skip_nils or meaning!='--NME--':
                                interpretations[form].add(meaning)
                        if meaning!='--NME--':
                                references[meaning].add(form)
        return interpretations, references

def prepare_lists(dist1, dist2):
	"""
	Create two aligned lists suitable to be plotted (on X and Y axes), based on two dictionaries that share keys.
	"""
	x_dist = []
	y_dist = []
	for i, freq in dist1.items():
		if i not in dist2:
			continue
		x_dist.append(freq)
		y_dist.append(dist2[i])
	x_dist=np.array(x_dist)
	y_dist=np.array(y_dist)
	return x_dist, y_dist


def get_freq_totals(articles, ambiguous_forms, skip_nils=True, ambiguous_only=True):
    """
    For all forms, get a frequency distribution of their meanings.
    """
    total_per_form=defaultdict(dict)
    for article in articles:
        for mention in article.entity_mentions:
                form=mention.mention
                meaning=mention.gold_link
                if skip_nils and meaning=='--NME--':
                    continue

                if ambiguous_only and form not in ambiguous_forms:
                    continue
                if meaning not in total_per_form[form]:
                    total_per_form[form][meaning]=0
                total_per_form[form][meaning]+=1
    return total_per_form

def get_pr_totals(articles, ambiguous_forms, uri_pr, skip_nils, ambiguous_only):
    """
    For all forms, get a PageRank distribution of their meanings.
    """
    total_per_form=get_freq_totals(articles, ambiguous_forms, skip_nils, ambiguous_only)
    form_pageranks=defaultdict(dict)
    for form, meanings in total_per_form.items():
        if ambiguous_only and form not in ambiguous_forms:
            continue
        #for uri, total in meanings.items():
            #acc_per_form_meaning[system][form][uri]=correct_per_form[form][uri]/total
        for uri in meanings.keys():
            if uri in uri_pr:
                form_pageranks[form][uri]=uri_pr[uri]
    return form_pageranks

def prepare_ranks(articles, ambiguous_forms, uri_pr=None, min_frequency=0, ambiguous_only=True, skip_nils=True, factor='freq'):
    """
    Group all form-instance pairs into ranks, based on the relative frequency of instances for a form.
    """
    rank_tuples=defaultdict(set)
    if factor == 'freq':
        total_per_form=get_freq_totals(articles, ambiguous_forms, skip_nils, ambiguous_only)
    else: # pagerank
        total_per_form=get_pr_totals(articles, ambiguous_forms, uri_pr, skip_nils, ambiguous_only)
    
    for form, data in total_per_form.items():
        if sum(data.values())<=min_frequency:
            continue
        sorted_by_rank=sorted(data.items(), key=lambda x:x[1], reverse=True)
        rank=1
        for ranked_URI, freq in sorted_by_rank:
            t=(form, ranked_URI)
            rank_tuples[rank].add(t)
            rank+=1

    return rank_tuples

def counts_to_log_counts(forms_by_count):
    """
    Convert the counts per form to their log values.
    """
    forms_by_log_count={}
    for count, forms in forms_by_count.items():
        try:
            log_count=math.log(count)
            forms_by_log_count[log_count] = forms
        except:
            print(count)
    return forms_by_log_count

def prepare_scores_to_plot(all_sys_accs):
    """
    Convert the performance scores to a list of lists, as needed to create a bar plot.
    """
    scores=[]
    for system, data in all_sys_accs.items():
        for evaluation, score in data.items():
            scores.append([evaluation, system, score])
    return scores

def get_freq_intervals(forms_by_count):
    """
    Obtain the frequency intervals (min, max) for each frequency bucket.
    """
    freqs_per_bucket=defaultdict(set)
    for count, forms in forms_by_count.items():
        try:
            log_count=math.log(count)
            rounded_log_count=round(log_count)
            freqs_per_bucket[rounded_log_count].add(count)
        except:
            print(count)
            
    interval_per_bucket = {}
    for freq, counts in freqs_per_bucket.items():
        mn = min(counts)
        mx = max(counts)
        interval_per_bucket[freq] = (mn,mx)
    return interval_per_bucket

def compute_counts_by_form(articles, skip_nils=True):
    """
    Compute counts by form. 
    """
    total_by_form = defaultdict(int)
    forms_by_count=defaultdict(set)

    for article in articles:
        for entity in article.entity_mentions:
            if entity.sys_link and (not skip_nils or entity.gold_link!='--NME--'):
                total_by_form[entity.mention]+=1

    for form, count in total_by_form.items():
        forms_by_count[count].add(form)

    return forms_by_count

################# RETRIEVAL AND ANALYSIS DONE ########################

################# EVALUATION #########################################

def overall_performance_prf(articles, skip_nils=True, skip_nonnils=False):
        """
        Compute overall precision, recall and F1 of a system.
        """
        tp=0
        fn=0
        fp=0
        for article in articles:
                for entity in article.entity_mentions:
                        if skip_nils and entity.gold_link=='--NME--':
                                continue
                        if skip_nonnils and entity.gold_link!='--NME--':
                                continue
                        if entity.gold_link==entity.sys_link:
                                tp+=1
                        else:
                                if entity.sys_link!='--NME--':
                                        fp+=1
                                if entity.gold_link!='--NME--':
                                        fn+=1
        print(tp, fp, fn)
        p=tp/(tp+fp)            
        r=tp/(tp+fn)            
        f1=2*p*r/(p+r)
        print(p,r,f1)
        return f1
        
def compute_prf_on_selection(arts, forms_set):
    """
    Compute precision, recall and F1 of a system on a predefined subset of all forms.
    """
    tp=0
    fn=0
    fp=0
    for article in arts:
        for entity in article.entity_mentions:
            if entity.mention in forms_set:
                if entity.gold_link==entity.sys_link:
                    tp+=1
                else:
                    if entity.sys_link!='--NME--':
                        fp+=1
                    if entity.gold_link!='--NME--':
                        fn+=1
    print(tp, fp, fn)
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    print(p,r,f1)
    return f1

def evaluate_ranks(articles, rank_tuples):
    """
    Evaluate ranks for precision, recall, and F1-score.
    """
    rank_tp=defaultdict(int)
    rank_fn=defaultdict(int)
    rank_fp=defaultdict(int)
    
    for article in articles:
        for mention in article.entity_mentions:
                form=mention.mention
                meaning=mention.gold_link
                sys_meaning=mention.sys_link
                t_gold=(form, meaning)
                t_sys=(form, sys_meaning)
                for rank, r_tuples in rank_tuples.items():
                    if t_gold in r_tuples and t_sys in r_tuples:
                        rank_tp[rank]+=1
                        break
                    elif t_gold in r_tuples:
                        rank_fn[rank]+=1
                    elif t_sys in r_tuples:
                        rank_fp[rank]+=1
    print('tp', rank_tp)
    print('fp', rank_fp)
    print('fn', rank_fn)
    
    rank_prec={}
    rank_recall={}
    rank_f1={}
    
    for rank in range(1,13):
        if rank_tp[rank]+rank_fp[rank]>0:
            rank_prec[rank]=rank_tp[rank]/(rank_tp[rank]+rank_fp[rank])
        else:
            rank_prec[rank]=0.0
        if rank_tp[rank]+rank_fn[rank]>0:
            rank_recall[rank]=rank_tp[rank]/(rank_tp[rank]+rank_fn[rank])
        else:
            rank_recall[rank]=0.0
        if rank_prec[rank]+rank_recall[rank]>0:
            rank_f1[rank]=2*rank_prec[rank]*rank_recall[rank]/(rank_prec[rank]+rank_recall[rank])
        else:
            rank_f1[rank]=0.0
    print('precision', rank_prec)
    print()
    print('recall', rank_recall)
    print()
    print('f1', rank_f1)
    print()
    return rank_prec, rank_recall, rank_f1

############## EVALUATION DONE #####################

############## DEPRECATED ###########################

def overall_performance(articles, skip_nils=True, skip_nonnils=False):
        """
        Compute overall accuracy of systems. Deprecated because now we use a more informative measure of precision, recall, and F1-score.
        """
        correct=0
        total=0
        for article in articles:
                for entity in article.entity_mentions:
                        if skip_nils and entity.gold_link=='--NME--':
                                continue
                        if skip_nonnils and entity.gold_link!='--NME--':
                                continue
                        if entity.gold_link==entity.sys_link:
                                correct+=1
                        total+=1
        print(correct, total)
        return correct/total

def compute_acc_on_selection(arts, forms_set):
    """
    Compute accuracy of a system on a subset of all forms. Deprecated because now we use a more informative measure of precision, recall, and F1-score.
    """
    correct=0
    total=0
    for article in arts:
        for entity in article.entity_mentions:
            if entity.mention in forms_set:
                total+=1
                if entity.gold_link==entity.sys_link:
                    correct+=1
    print(correct, total)
    return correct/total

def compute_accuracy_by_form(articles, skip_nils=True):
    """
    Compute accuracy by form. Deprecated: now we compute precision, recall, and F1-score.
    """
    forms_acc = defaultdict(int)

    correct_by_form = defaultdict(int)
    total_by_form = defaultdict(int)

    forms_by_count=defaultdict(set)

    for article in articles:
        for entity in article.entity_mentions:
            if entity.sys_link and (not skip_nils or entity.gold_link!='--NME--'):
                if entity.sys_link==entity.gold_link:
                    correct_by_form[entity.mention]+=1
                total_by_form[entity.mention]+=1

    for form, count in total_by_form.items():
        forms_by_count[count].add(form)
        forms_acc[form]=correct_by_form[form]*100.0/count

    return forms_acc, forms_by_count

def compute_accuracy_by_uri(articles, skip_nils=True):
        """
        Compute accuracy by URI. Deprecated: now we compute precision, recall, and F1-score.
        """
        uris_acc = defaultdict(int)
        correct_by_uri = defaultdict(int)
        total_by_uri = defaultdict(int)

        uris_by_count = defaultdict(set)

        for article in articles:
                for entity in article.entity_mentions:
                        if entity.sys_link and (not skip_nils or entity.gold_link!='--NME--'):
                                if entity.sys_link==entity.gold_link:
                                        correct_by_uri[entity.gold_link]+=1
                                total_by_uri[entity.gold_link]+=1
        for uri, count in total_by_uri.items():
                uris_by_count[count].add(uri)
                uris_acc[uri]=correct_by_uri[uri]*100.0/count

        return uris_acc, uris_by_count

###############################################
