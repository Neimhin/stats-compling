import pandas as pd
from functools import reduce
import re
import nltk
import string
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib as mpl
sns.set_theme()
mpl.rcParams["text.usetex"]=True
mpl.rcParams["lines.linewidth"]=0.9
mpl.rcParams["figure.dpi"]=400

syn_proxies = [
  'inverse mean-node-degree',
  'max-tree-depth',

  'node-type-diversity',
  'num-nodes',
  'num-node-types',

  'sentence-length-bare',
  'sentence-length',
]

lex_proxies_dirty = [
  'mean-word-length',
  'lexical-diversity words',
  'lexical-density words',
  'lexical-density lemmas',
  'lexical-diversity lemmas',
  'TTR words',
  'TTR lemmas',
]

lex_proxies_clean = [
  'lexical-diversity words-clean',
  'lexical-density words-clean',
  'lexical-density lemmas-clean',
  'lexical-diversity lemmas-clean',
  'TTR words-clean',
  'TTR lemmas-clean',
]

lex_proxies_bare = [
  'mean-word-length words-clean-bare',
  'TTR words-clean-bare',
  'TTR lemmas-clean-bare',
  'lexical-diversity words-clean-bare',
  'lexical-density words-clean-bare',
  'lexical-density lemmas-clean-bare',
  'lexical-diversity lemmas-clean-bare',
]

lex_proxies = list(set([*lex_proxies_dirty,*lex_proxies_clean,*lex_proxies_bare]))

proxies = syn_proxies + lex_proxies
proxies_dirty = syn_proxies + lex_proxies_dirty
proxies_clean = syn_proxies + lex_proxies_clean
proxies_bare = syn_proxies + lex_proxies_bare


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def rank_forums_by_proxy_fig(df, t):
    if t == "syn":
      prox = syn_proxies
    elif t == "lex":
      prox = lex_proxies_dirty
    elif t == "all":
      prox = proxies
    prox = sorted(prox)

    mean_values = df.groupby("forum")[prox].mean()
    ranked_forums = mean_values.rank(ascending=False).astype(int)

    print("Forum Rankings by Proxy Measures:")
    print(ranked_forums.to_string())

    plt.clf()
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(ranked_forums, annot=True, cmap=cmap(), linewidths=.5, fmt='d', cbar=False)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.title("Forum Rankings by Proxy Measures")
    plt.tight_layout()
    plt.savefig("heatmap-rank-fora-by-proxies." +t+".png")


def rank_forums_by_proxy(df, proxies):
    mean_values = df.groupby("forum")[proxies].mean()
    ranked_forums = mean_values.rank(ascending=False).astype(int)

    print("Forum Rankings by Proxy Measures:")
    print(ranked_forums.to_string())

def descriptive_statistics(df, proxies):
    import pandas as pd
    # Compute the descriptive statistics for the entire dataframe
    entire_frame_stats = df[proxies].describe().transpose()
    print("Entire Frame:")
    print(entire_frame_stats.to_string())

    # Compute the descriptive statistics for each forum
    print("\nPer Forum:")
    for forum in df["forum"].unique():
        forum_df = df[df["forum"] == forum]
        forum_stats = forum_df[proxies].describe().transpose()
        print(f"\n{forum} Forum:")
        print(forum_stats.to_string())

def compare_complexity_nonparametric(df, proxies, category_col, p_value_threshold=0.05):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from statsmodels.stats.multicomp import MultiComparison
    results = {}
    for proxy in proxies:
        # Perform Kruskal-Wallis H-test
        categories = df[category_col].unique()
        data = [df[df[category_col] == category][proxy] for category in categories]
        h_stat, p_value = stats.kruskal(*data)
        # Perform post-hoc Dunn's test with Bonferroni correction
        if p_value < p_value_threshold:
            mc = MultiComparison(df[proxy], df[category_col])
            result = mc.allpairtest(stats.mannwhitneyu, method='bonf')[0]
            results[proxy] = {
                'h_stat': h_stat,
                'p_value': p_value,
                'dunn_result': result
            }
        else:
            results[proxy] = {
                'h_stat': h_stat,
                'p_value': p_value,
                'dunn_result': None
            }
    return results


def make_correlation_plots(df):
  import matplotlib.pyplot as plt
  for codist in [True, False]:
    corrcoefs(df, proxies=proxies, codist=codist); plt.savefig(f"corr-plot-all-{codist}.png")
    corrcoefs(df, proxies=proxies_dirty,codist=codist); plt.savefig(f"corr-plot-dirty-{codist}.png")
    corrcoefs(df, proxies=proxies_clean,codist=codist); plt.savefig(f"corr-plot-clean-{codist}.png")
    corrcoefs(df, proxies=proxies_bare,codist=codist); plt.savefig(f"corr-plot-bare-{codist}.png")


def pearson_spearman(x,y):
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    if x is y:
        return "same"
    r, p = pearsonr(x, y)
    spearman, pspearman = spearmanr(x, y)
    return {"pearson-r": r, "pearson-p": p, "spearman-r": spearman, "spearman-p": pspearman}

def correlations(df,proxies=proxies):
    return {p: {p2: pearson_spearman(df[p],df[p2]) for p2 in proxies} for p in proxies}

def boxplot(df,y, x="forum"):
  import matplotlib.pyplot as plt
  plt.clf()
  fig = sns.boxplot(
    x=x,
    y=y,
    data=df).get_figure()
  fig.tight_layout()
  fig.savefig("boxplot-" + y + ".png")

def corrcoefs(x,forum=r".",proxies=proxies, codist=True):
    import matplotlib.pyplot as plt
    from scipy.cluster import hierarchy
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    plt.clf()
    
    cm = cmap()
    
    x = x[x["forum"].str.contains(forum)]
    fora = x["forum"].unique()
    x = x[proxies]
    x = x.dropna(axis=0)
    print(fora)
    print(x)
    def reg_coef(x, y, label=None, color=None, **kwargs):
        ax = plt.gca()
        r, p = pearsonr(x, y)
        spearman, pspearman = spearmanr(x, y)
        
        # Create boxes with labels and set their facecolors
        pearson_bbox_props = dict(boxstyle="round,pad=0.3", fc=cm((r + 1.0) / 2.0), ec="none")
        spearman_bbox_props = dict(boxstyle="round,pad=0.3", fc=cm((spearman + 1.0) / 2.0), ec="none")
        
        ax.annotate(
        "Pearson\'s r = {:.2f}".format(r),
        xy=(0.3, 0.4),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=pearson_bbox_props,
        )
        
        ax.annotate(
        "Spearman\'s r = {:.2f}".format(spearman),
        xy=(0.3, 0.6),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=spearman_bbox_props,
        )
        
        # Set axis off
        ax.set_axis_off()
    
    # def hist(x,label=None,color=None,**kwargs):
    #   return sns.histplot(x,label=label,color=color,kde=True)

    def hist(x, label=None, color=None,sample=50,  **kwargs):
        # Test for normality using Shapiro-Wilk test
        stat, p_value = shapiro(x,sample=50)
        # Add the p-value to the histogram plot
        ax = plt.gca()
        ax.annotate("Shapiro-Wilk\np-value = {:.2e}\nN = {}".format(p_value,sample), xy=(0.2, 0.9), xycoords="axes fraction", fontsize=10)

        return sns.histplot(x, label=label, color=color, kde=True)

    
    g = sns.PairGrid(x)
    plt.suptitle(' '.join([f for f in fora]))
    g.map_diag(hist)
    if codist:
      g.map_upper(sns.scatterplot)
    g.map_lower(reg_coef)
    g.tight_layout()
    return g

# def ks(x):
#     from scipy.stats import kstest
#     # Take a random sample of 100 data points
# 
#     # Test for normality using Shapiro-Wilk test
#     stat, p_value = kstest(x)
#     return (stat, p_value)
# 
# def kses(df,proxies=proxies):
#     return { p: ks(df[p]) for p in proxies }

def shapiro(x,sample=50):
    from scipy.stats import shapiro
    # Take a random sample of 100 data points
    samples = x.sample(n=min(sample, len(x)), random_state=42)

    # Test for normality using Shapiro-Wilk test
    stat, p_value = shapiro(samples)
    return (stat, p_value)

def shapiros(df,proxies=proxies):
    return { p: shapiro(df[p]) for p in proxies }


def perform_chi2_test(df, proxies, forum_col="forum", categories=("twitter", "speech")):
    import pandas as pd
    from scipy.stats import chi2_contingency

    results = {}
    for measure in proxies:
        # Filter data by the two categories
        cat1_data = df[df[forum_col] == categories[0]][measure]
        cat2_data = df[df[forum_col] == categories[1]][measure]

        print(cat1_data)
        print(cat2_data)
        # Calculate contingency table
        contingency_table = pd.crosstab(cat1_data, cat2_data)
        print(contingency_table)
        # Perform chi2 test
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        # Store results
        results[measure] = {"chi2": chi2, "p_value": p_value}
    return results

import pandas as pd
from scipy.stats import f_oneway

def perform_anova_test(df, proxies, forum_col="forum", categories=("twitter", "speech")):
    results = {}
    
    for measure in proxies:
        results[measure] = {}
        
        # Filter data by the categories
        cat_data = [df[df[forum_col].str.contains(cat)][measure] for cat in categories]
        
        # Perform ANOVA test
        f_stat, p_value = f_oneway(*cat_data)
        
        # Store results
        results[measure] = {"f_stat": f_stat, "p_value": p_value}
    
    return results


def is_question_number_ending(row):
  if row["forum"] != "question-written" and row["forum"] != "question-oral":
    return False
  lemmas = row["lemmas"]
  def match(t):
    return is_question_number_ending.num_re.match(t)
  if lemmas[0] == "[" and lemmas[2] == "]" and match(lemmas[1]):
    return True
  return False
is_question_number_ending.num_re = re.compile(r"\d+/\d+")


def apply_lexical_measures(df, big):
  ss = ["words","words-clean", "words-clean-bare", "lemmas", "lemmas-clean","lemmas-clean-bare"]
  for t in ss:
    df[f"lexical-density {t}"] = lexical_density(big[t])
    df[f"lexical-diversity {t}"] = lexical_diversity(big[t])
  return df

def load_proxies():
    return pd.read_csv("proxies.csv",low_memory=False)

def save_proxies(df):
  df.to_csv("proxies.csv")

def load_df():
  return pd.read_pickle("d.pickle")


def drop_non_serial(df):
  return df.copy().drop(["parse","parse-sentences"],axis=1)

def copy_proxies(df):
  cols = [ 'id', 
      'sentence_no',
      'name',
      'party',
      'age in days',
      'forum',
      'language',
      'date',
      'topic',
      'cardinal number in debate',
      'TTR words',
      'TTR lemmas',
      'TTR lemmas clean',
      'TTR lemmas clean and bare',
      'mean-node-degree',
      'max tree depth',
      'num_node_types',
      'node-type-diversity',
      'num-nodes',
      'num-node-types',
      'sentence-length-bare',
      'sentence-length',
      'mean-word-length',
      'lexical-density-words',
      'lexical-density-lemmas',
      'lexical-density-lemmas-clean',
      'lexical-density-lemmas-clean-bare'
    ]
  return df[cols].copy()

def lexical_density(series):
  import numpy as np
  t2l = token2lexscore(series)
  return series.apply(lambda sent: [t2l[word] for word in set(sent)]).apply(np.mean)

def lexical_diversity(series):
  import numpy as np
  t2l = token2lexscore(series)
  return series.apply(lambda sent: [t2l[word] for word in set(sent)]).apply(np.nansum)
  #return series.apply(lambda sent: np.nansum([t2l[word] for word in set(sent)]))
  

def token2lexscore(series):
  from gensim import corpora
  import numpy as np
  dct = corpora.Dictionary(doc for doc in series)
  
  df = pd.DataFrame([(k,dct.cfs[v]) for k,v in dct.token2id.items()],columns=["token","frequency"])
  
  df = df.sort_values("frequency")
  
  df["rank"] = df["frequency"].rank(method="dense",ascending=False)
  df["lexscore"] = np.log2(df["rank"]+1)
  return {
    tok: score for tok,score in zip(df["token"],df["lexscore"])
  } 

def save_df(df):
  df.to_pickle("d.pickle")

def term_frequencies(document):
  from gensim import corpora
  dct = corpora.Dictionary([word] for word in document)  
  return [dct.cfs[dct.token2id[word]] for word in document]

def lexicon_frequency(series):
  from gensim import corpora
  dictionary = corpora.Dictionary(doc for doc in series)
  return dictionary

def dotfidfs(documents):
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  # Custom tokenizer that skips tokenization
  def identity_tokenizer(text):
      return text
  
  # Create an instance of TfidfVectorizer with the custom tokenizer
  vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, token_pattern=None)
  
  # Fit and transform the documents into a term-document matrix
  tdm = vectorizer.fit_transform(documents)
  
  # Get the feature names
  feature_names = vectorizer.get_feature_names_out()
  
  # Function to extract the TF-IDF scores for a document
  def extract_tfidf_scores(document_idx):
      return {
          feature_names[word_idx]: tdm[document_idx, word_idx] for word_idx in range(tdm[document_idx].shape[1])
      }
  
  # Create lists of TF-IDF scores for each document
  tfidf_scores_lists = [
      [extract_tfidf_scores(doc_idx).get(word, 0) for word in doc] for doc_idx, doc in enumerate(documents)
  ]
  return pd.Series(tfidf_scores_list)  


def tdm_df(documents):
  import pandas as pd
  from sklearn.feature_extraction.text import TfidfVectorizer
  def identity_tokenizer(text):
      return text
  vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, token_pattern=None)
  tdm = vectorizer.fit_transform(documents)
  feature_names = vectorizer.get_feature_names()
  tdm_df = pd.DataFrame(tdm.toarray(), columns=feature_names)
  
  return [ tdms[word] for (word, (i,tdms)) in list(zip(documents,tdm_df.iterrows())) ]

at_mention_re = re.compile(r'@\w+')
hashtag_re = re.compile(r'#\w+')
url_re = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
punct_re = re.compile(r'^[^\w\s]+$')
special_re = re.compile(r'__[A-Z]+__')

sub = {
        "at":         lambda x: at_mention_re.sub("__ATMENTION__",x),
        "hashtag":    lambda x: hashtag_re.sub("__HASHTAG__",x),
        "url":        lambda x: url_re.sub("__URL__",x),
        "punct":      lambda x: punct_re.sub("__PUNCT__",x),
}

def count_specials(series):
  reg = {
          "at":         re.compile(r"__ATMENTION__"),
          "hashtag":    re.compile(r"__HASHTAG__"),
          "url":        re.compile(r"__URL__"),
          "punct":      re.compile(r"__PUNCT__"),
  }
  d = {}
  for t in reg:
    count = series.apply(lambda x: int(not not reg[t].match(x))).sum() 
    d[t] = (count, count/len(series)) 
  import numpy as np
  d["total"] = np.sum([d[t][0] for t in d])
  d["total"] = (d["total"],d["total"]/len(series))
  return d

def fora(df):
    import re
    unique_categories = df['forum'].unique()
    for forum in ["twitter","speech","question","answer"]:
        reg = re.compile(forum)
        yield (forum,df[df["forum"].apply(reg.match).apply(bool)])


def remove_special(series):
  def remove_special_token(token):
    if special_re.match(token):
      return None
    return token
  def remove_specials(tokens):
    return list(filter(lambda i: i is not None, [remove_special_token(token) for token in tokens]))
  return series.apply(remove_specials)

def clean_word(word):
    word = word.lower()
    for t in sub:
        a = sub[t](word)
        if word is not a:
            return a
    return word

def clean_sentence(s):
    return [clean_word(w) for w in s]

def clean(series):
    return series.apply(clean_sentence)


def lowercase(df, text='text'):
    df[text] = df[text].apply(lambda x: x.lower())
    return df

def remove_at_mentions(df, text='text'):
    df[text] = df[text].apply(lambda x: re.sub(r'@\w+', '__ATMENTION__', x))
    return df

def remove_hashtags(df, text='text'):
    df[text] = df[text].apply(lambda x: re.sub(r'#\w+', '__HASHTAG__', x))
    return df


def remove_urls(df, text='text'):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    df[text] = df[text].apply(lambda x: url_pattern.sub('__URL__', x))
    return df

def whitespace(df, text='text'):
    pattern1 = re.compile(r'\s+')
    pattern2 = re.compile(r'(^\s+)|(\s+$)')
    df[text] = df[text].apply(lambda x: pattern1.sub(' ',x).strip())
    return df

def punctuation(df, text="text"):
    import unicodedata
    punctuation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
    def simple_punctuation(t):
      return ''.join((x if unicodedata.category(x) not in punctuation_cats or x == "_" else "!") for x in t)
    punct = string.punctuation.replace("_","")
    tra = "".maketrans(punct, '\t'*len(punct))
    df[text] = df[text].apply(lambda x: simple_punctuation(x).translate(tra).replace(r"\t+", " __PUNCT__ "))
    return df


def preprocess(df, text='text'):
    df = lowercase(df, text)
    df = remove_hashtags(df, text)
    df = remove_at_mentions(df, text)
    df = remove_urls(df, text)
    df = punctuation(df, text)
    df = whitespace(df, text)
    return df

def ordering(df):
  ordering=[df.columns[i] for i in [4, 1, 3, 0, 2, 5, 6]]
def reorder(df):
  from scipy.cluster import hierarchy
  ordering=[df.columns[i] for i in [4, 1, 3, 0, 2, 5, 6]]
  print(ordering)
  return df[ordering]

def cmap():
  import seaborn as sns
  cmap = sns.diverging_palette(h_neg=220,h_pos=45,s=74,l=73,sep=10,n=14,center="light",as_cmap=True)
  return cmap

def calculate_ttr(tokens):
  from collections import Counter
  types = Counter(tokens)
  ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
  return ttr

def isTerminal(pt):
  return len(pt.children) == 0

def isPenTerminal(pt):
  return len(pt.children) == 1 and len(pt.children[0].children) == 0

def max_tree_depth(pt):
  if len(pt.children) == 0:
    return 0
  return max(list(map(lambda x: x+1,map(max_tree_depth,pt.children))))

def node2string(node):
  return label(node) + "_" + "_".join([label(n) for n in node.children])

def node_degree(node):
  return len(node.children)

def load_parses():
  return pd.read_pickle("data/parses.pickle")

def load(s):
  return pd.read_csv(s,delimiter="\t")

def saveid2x(df,s,f):
  df[["id",s]].to_csv(f,sep="\t",index=False)

def sentence_node_degree_list(s):
  return [node_degree(s)] + reduce(lambda a,x:a+x, list([sentence_node_degree_list(sub) for sub in s.children if not isPenTerminal(sub)]),[])

def sentence_node_type_list(s):
  return [node2string(s)] + reduce(lambda a,x:a+x, list(filter(None,[sentence_node_type_list(sub) for sub in s.children if not isPenTerminal(sub)])),[])

def label(node):
  if isTerminal(node):
    return "TERMINAL"
  return node.label

def mean_word_length(sent):
  import numpy as np
  return np.mean([len(word) for word in sent])
