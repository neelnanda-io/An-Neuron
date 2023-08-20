# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)


model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-large")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
utils.test_prompt("I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked", "an", model)
# %%
data = load_dataset("stas/openwebtext-10k", split="train")
strings = [i for i in data["text"] if len(i)>2000]
len(strings)
# %%
num_prompts = 64
n_ctx = 256
tokens = model.to_tokens(strings[num_prompts:2*num_prompts])[:, :n_ctx]
tokens.shape
# %%
logits = model(tokens)
A = model.to_single_token(" a")
AN = model.to_single_token(" an")
logit_diff = logits[:, :, AN] - logits[:, :, A]
logit_diff = torch.where(tokens==AN, torch.zeros_like(logit_diff)-5, logit_diff)
# histogram(logit_diff.flatten())
# %%
token_df = pd.DataFrame({"logit_diff": to_numpy(logit_diff.flatten()), "batch": [b for b in range(num_prompts) for p in range(n_ctx)], "pos": [p for b in range(num_prompts) for p in range(n_ctx)]})
nutils.show_df(token_df.sort_values("logit_diff", ascending=False).head(100))
# %%
temp_df = token_df.sort_values("logit_diff", ascending=False).head(20)
import circuitsvis as cv
for row in temp_df.iterrows():
    b = int(row[1]["batch"])
    p = int(row[1]["pos"])
    print(b, p)
    str_toks = model.to_str_tokens(tokens[b, :p+3])
    vals = logit_diff[b, :p+3]
    nutils.create_html(str_toks, vals)
    # display(cv.tokens.colored_tokens(str_toks, vals))

# %%
utils.test_prompt("Ross Perot, who ran as", "an", model)
utils.test_prompt("Van der Bellen, who ran as", "an", model)
utils.test_prompt("Senator Angus King, who ran as", "an", model)
utils.test_prompt("Spot gold was up 0.7 percent at $1,340.90", "an", model)
utils.test_prompt("An eye for", "an", model)
utils.test_prompt("Every action has", "an", model)
utils.test_prompt("I'll be there in under half", "an", model)
utils.test_prompt("I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked", "an", model)
# %%
prompts = [
"Ross Perot, who ran as",
"Van der Bellen, who ran as",
"Senator Angus King, who ran as",
"Spot gold was up 0.7 percent at $1,340.90",
"An eye for",
"Every action has",
"I'll be there in under half",
"I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked",
]

tokens = model.to_tokens(prompts)
final_index = (tokens!=model.tokenizer.pad_token_id).sum(-1)
tokens[np.arange(len(prompts)), final_index]
# tokens[np.arange(len(prompts)), final_index+1]
# %%
unembed_dir = model.W_U[:, AN] - model.W_U[:, A]
logits, cache = model.run_with_cache(model.to_tokens(prompts))
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, return_labels=True)
final_resid_stack = resid_stack[:, np.arange(len(prompts)), final_index, :]
(final_resid_stack @ unembed_dir).shape


# %%
# line((final_resid_stack @ unembed_dir).T, line_labels=[p[:10] for p in prompts], x=resid_labels, title="Direct Logit Diff Attribution")
# %%
final_ln_scale = cache["scale"][np.arange(len(prompts)), final_index, 0]
print(final_ln_scale.shape)
for layer in [31, 34, 35]:
    neuron_acts = cache["post", layer][np.arange(len(prompts)), final_index, :]
    neuron_wdla = (model.blocks[layer].mlp.W_out @ unembed_dir) / final_ln_scale[:, None]
    line(neuron_acts * neuron_wdla, line_labels=[p[:10] for p in prompts], title=f"Neuron DLA in L{layer}")
# %%
random_normal = torch.randn(d_mlp, d_mlp)
random_subspace = random_normal.qr()[0]
random_subspace.shape
# %%
layer = 31
neuron_acts = cache["post", layer][np.arange(len(prompts)), final_index, :]
neuron_wdla = (model.blocks[layer].mlp.W_out @ unembed_dir) / final_ln_scale[:, None]
# line((neuron_acts * neuron_wdla) @ random_subspace.cuda(), line_labels=[p[:10] for p in prompts], title=f"Neuron DLA in L{layer} under random rotation")
# line((neuron_acts * neuron_wdla), line_labels=[p[:10] for p in prompts], title=f"Neuron DLA in L{layer} (incl LN scale)")
# %%
ni = 892
win = model.blocks[layer].mlp.W_in[:, ni]
wout = model.blocks[layer].mlp.W_out[ni, :]
resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer, mlp_input=True, expand_neurons=False, apply_ln=True, return_labels=True)
neuron_resid_stack = resid_stack[:, np.arange(len(prompts)), final_index, :]
# (neuron_resid_stack @ unembed_dir).shape

# line((neuron_resid_stack @ win).T, line_labels=[p[:10] for p in prompts], x=resid_labels, title="Direct Logit Diff Attribution")

# %%
layer2 = 23
l31_ln_scale = cache["scale", layer, "ln2"][np.arange(len(prompts)), final_index, 0]
prev_neuron_acts = cache["post", layer2][np.arange(len(prompts)), final_index, :]
prev_neuron_wdla = (model.blocks[layer2].mlp.W_out @ win) / l31_ln_scale[:, None]
# line((prev_neuron_acts * prev_neuron_wdla) @ random_subspace.cuda(), line_labels=[p[:10] for p in prompts], title=f"Neuron DLA in L{layer} under random rotation")
# line((prev_neuron_acts * prev_neuron_wdla), line_labels=[p[:10] for p in prompts], title=f"L{layer2} effect on neuron L31N892")
# line(prev_neuron_wdla, line_labels=[p[:10] for p in prompts], title=f"L{layer2} weight-based effect on neuron L31N892")

# %%
ni2 = 4377
win = model.blocks[layer2].mlp.W_in[:, ni2]
wout = model.blocks[layer2].mlp.W_out[ni2, :]
resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer2, mlp_input=True, expand_neurons=False, apply_ln=True, return_labels=True)
neuron_resid_stack = resid_stack[:, np.arange(len(prompts)), final_index, :]
# (neuron_resid_stack @ unembed_dir).shape

# line((neuron_resid_stack @ win).T, line_labels=[p[:10] for p in prompts], x=resid_labels, title=f"Direct Linear Attribution on neuron L{layer2}N{ni2}")

# %%
full_vocab = model.to_str_tokens(np.arange(d_vocab))
category = []
for i, token in enumerate(full_vocab):
    if token.startswith(" ") and token !=" ":
        if token[1].isalpha():
            if token[1].lower() in "aeiou":
                category.append("vowel")
            else:
                category.append("not_vowel")
        else:
            category.append("not_word")
    else:
        category.append("not_word")
vocab_df = pd.DataFrame({"token": full_vocab, "category": category, "ind": np.arange(d_vocab)})
vocab_df
# %%
vowel_embeds = model.W_E[(vocab_df["category"]=="vowel").values]
not_vowel_embeds = model.W_E[(vocab_df["category"]=="not_vowel").values]

embeds = torch.cat([vowel_embeds, not_vowel_embeds], dim=0)
labels = np.array([1]*len(vowel_embeds) + [0]*len(not_vowel_embeds))

from sklearn.linear_model import LogisticRegression
probe = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(to_numpy(embeds), labels, test_size=0.2, random_state=42)
probe.fit(X_train, y_train)

y_pred = probe.predict(X_test)
print((y_pred==y_test).astype(np.float32).mean())
# %%
temp_df = pd.DataFrame(dict(
    y = X_test @ probe.coef_[0],
    label=[["consonant", "vowel"][i] for i in y_test],
))
# px.histogram(temp_df, x="y", color="label", barmode="overlay", histnorm="percent", title="Embeddings are linearly separable by vowelness")
# %%
layer3 = 26
head = 1
pos = 16
prompt_index = -1
attn_layer = model.blocks[layer3].attn
win = model.blocks[layer].mlp.W_in[:, ni]
win_via_head = attn_layer.W_V[head] @ attn_layer.W_O[head] @ win
resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer3, mlp_input=False, expand_neurons=False, apply_ln=True, return_labels=True)
neuron_resid_stack = resid_stack[:, -1, pos, :]
# (neuron_resid_stack @ unembed_dir).shape

# line((neuron_resid_stack @ win_via_head), x=resid_labels, title=f"Linear Attribution on neuron L{layer}N{ni} via head L{layer3}H{head}")
# %%
l31_ln_scale = cache["scale", layer, "ln2"][np.arange(len(prompts)), final_index, 0]
# prev_neuron_acts = cache["post", layer2][np.arange(len(prompts)), final_index, :]
# prev_neuron_wdla = (model.blocks[layer2].mlp.W_out @ win) / l31_ln_scale[:, None]
# # line((prev_neuron_acts * prev_neuron_wdla) @ random_subspace.cuda(), line_labels=[p[:10] for p in prompts], title=f"Neuron DLA in L{layer} under random rotation")
# line((prev_neuron_acts * prev_neuron_wdla), line_labels=[p[:10] for p in prompts], title=f"L{layer2} effect on neuron L31N892")
# line(prev_neuron_wdla, line_labels=[p[:10] for p in prompts], title=f"L{layer2} weight-based effect on neuron L31N892")

# %%
token_list = [f"{s}/{i}" for i, s in enumerate(model.to_str_tokens(prompts[-1]))]
layers = [15, 0, 7, 3, 1]
for temp_layer in layers:
    neuron_acts = cache["post", temp_layer][-1, pos, :]
    neuron_wdla = model.blocks[temp_layer].mlp.W_out @ win_via_head
    ln_scale = cache["scale", layer3, "ln1"][-1, 16, 0]
    line(neuron_acts * neuron_wdla / ln_scale, title=f"L{temp_layer} neurons effect on neuron L{layer}{ni} via head L{layer3}H{head}")
# %%
s = "Be there in under half an hour umbrella user underarm apple banana pear argument one useless unicorn herd heir honestly hourly happiness haggard helpful helicopter hobbit"
ni4 = 1595
layer4 = 0
temp_logits, temp_cache = model.run_with_cache(s)
temp_acts = to_numpy(temp_cache["post", layer4][0, :, ni4])
nutils.create_html(model.to_str_tokens(s), temp_acts)
ni5 = 4007
layer5 = 15
temp_logits, temp_cache = model.run_with_cache(s)
temp_acts = to_numpy(temp_cache["post", layer5][0, :, ni5])
nutils.create_html(model.to_str_tokens(s), temp_acts)
# %%
num_prompts = 64
n_ctx = 256
big_tokens = model.to_tokens(strings[num_prompts:2*num_prompts])[:, :n_ctx]
big_tokens.shape
big_logits, big_cache = model.run_with_cache(big_tokens, names_filter=lambda x: x.endswith("post") or "pattern" in x)
x = to_numpy(big_cache["post", layer4][:, :, ni4])
y = to_numpy(big_cache["post", layer5][:, :, ni5])
# px.scatter(x=x.flatten(), y=y.flatten(), labels=dict(x=f"L{layer4}N{ni4}", y=f"L{layer5}N{ni5}"), trendline="ols", title="Correlation between the two vowel neurons")
# %%
neuron_layers = [15, 15, 7, 0]
neuron_index = [4007, 2003, 2183, 1595]
for layer_n, index_n in zip(neuron_layers, neuron_index):
    resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=layer_n, mlp_input=True, expand_neurons=False, apply_ln=True, return_labels=True)
    neuron_resid_stack = resid_stack[:, -1, pos, :]

    line((neuron_resid_stack @ model.blocks[layer_n].mlp.W_in[:, index_n]), x=resid_labels, title=f"Linear Attribution on neuron L{layer_n}N{index_n}")

    y = to_numpy(big_cache["post", layer_n][:, :, index_n])
    # px.scatter(x=x.flatten(), y=y.flatten(), labels=dict(x=f"L{layer4}N{ni4}", y=f"L{layer_n}N{index_n}"), trendline="ols", title="Correlation between the two vowel neurons").show()
    print(cache["post", 0][-1, 16, 1595] * (model.blocks[0].mlp.W_out[1595, :] @ model.blocks[layer_n].mlp.W_in[:, index_n]) / cache["scale", layer_n, "ln2"][-1, 16, 0])


# %%
neuron_layers = [15, 7, 0]
neuron_index = [4007, 2183, 1595]
residual_norms = []
for layer_n, index_n in zip(neuron_layers, neuron_index):
    wout = model.blocks[layer_n].mlp.W_out[index_n, :]
    print(wout.norm())
    acts = big_cache["post", layer_n][:, :, index_n]
    resid_post = big_cache["resid_post", 15].norm(dim=-1)
    residual_norms.append(acts.flatten() * wout.norm() / resid_post.flatten())
residual_norms = torch.stack(residual_norms)
# histogram(residual_norms[:, einops.repeat(residual_norms.min(0).values>0.0, "n -> n")].T, barmode="overlay")
# %%
vecs = [
    model.blocks[31].mlp.W_out[892, :],
    model.blocks[31].mlp.W_in[:, 892],
    model.W_U[:, A],
    model.W_U[:, AN],
    model.W_U[:, AN] - model.W_U[:, A],
    ]
vecs = torch.stack(vecs)
vecs_norm = vecs / vecs.norm(dim=-1, keepdim=True)
vecs_labels = ['wout', 'win', 'a_unembed', 'an_unembed', 'diff_unembed']
imshow(vecs_norm @ vecs_norm.T, x=vecs_labels, y=vecs_labels, title="Cosine similarity between vectors for L31N892")
# %%
alg_prompts = ([
    "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked",
    "I climbed up the pear tree and picked a pear. I climbed up the orange tree and picked",
    "I climbed up the pear tree and picked a pear. I climbed up the elephant tree and picked",
    "I climbed up the pear tree and picked a pear. I climbed up the pineapple tree and picked",
    "I climbed up the pear tree and picked a pear. I climbed up the strawberry tree and picked",
    "I climbed up the pear tree and picked a pear. I climbed up the plum tree and picked",
])
alg_tokens = model.to_tokens(alg_prompts)
alg_logits, alg_cache = model.run_with_cache(alg_tokens)
# line(alg_logits[:, -1, AN] - alg_logits[:, -1, A])
# %%
imshow(alg_cache["pattern", 26][:, 1, :, :], facet_col=0, x=token_list, y=token_list)
# %%
fruit_labels = ["apple", "orange", "elephant", "pineapple", "strawberry", "plum"]
query = alg_cache["q", 26][:, -1, 1, :]
key = alg_cache["k", 26][:, 16, 1, :]
W_Q = model.blocks[26].attn.W_Q[1]
W_K = model.blocks[26].attn.W_K[1]
W_key_side = query @ W_K.T
W_query_side = key @ W_Q.T
resid_stack, resid_labels = alg_cache.get_full_resid_decomposition(layer=26, mlp_input=False, expand_neurons=False, apply_ln=True, pos_slice=16, return_labels=True)
# line((resid_stack * W_key_side[None, :, :]).sum(-1).T, line_labels=fruit_labels, x=resid_labels, title="Key side of attention")
resid_stack, resid_labels = alg_cache.get_full_resid_decomposition(layer=26, mlp_input=False, expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
# line((resid_stack * W_query_side[None, :, :]).sum(-1).T, line_labels=fruit_labels, x=resid_labels, title="Query side of attention")
# %%
# line((alg_cache["post", 16][:, 16, :, None] * model.blocks[16].mlp.W_out[None, :, :] * W_key_side[:, None, :]).sum(-1), line_labels=fruit_labels)
# %%
line(big_cache["pattern", 26][:, 1, :, 1:].max(-1).values)
# %%
neuron_df = nutils.make_neuron_df(n_layers, d_mlp)
W_out_norm = nutils.normalise(model.W_out)
W_in_norm = nutils.normalise(model.W_in, dim=-2)
unembed_dir_norm = nutils.normalise(model.W_U[:, AN] - model.W_U[:, A])
neuron_df["cos_an"] = to_numpy(W_out_norm @ unembed_dir_norm).flatten()
neuron_df["cos_an_in"] = to_numpy(unembed_dir_norm @ W_in_norm).flatten()
nutils.show_df(neuron_df.sort_values("cos_an", ascending=False).head(20))

# %%
# def list_flatten(nested_list):
#     return [x for y in nested_list for x in y]
# def make_token_df(tokens, len_prefix=5, len_suffix=1):
#     str_tokens = [model.to_str_tokens(t) for t in tokens]
#     unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]
    
#     context = []
#     batch = []
#     pos = []
#     label = []
#     for b in range(tokens.shape[0]):
#         # context.append([])
#         # batch.append([])
#         # pos.append([])
#         # label.append([])
#         for p in range(tokens.shape[1]):
#             prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
#             if p==tokens.shape[1]-1:
#                 suffix = ""
#             else:
#                 suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
#             current = str_tokens[b][p]
#             context.append(f"{prefix}|{current}|{suffix}")
#             batch.append(b)
#             pos.append(p)
#             label.append(f"{b}/{p}")
#     # print(len(batch), len(pos), len(context), len(label))
#     return pd.DataFrame(dict(
#         str_tokens=list_flatten(str_tokens),
#         unique_token=list_flatten(unique_token),
#         context=context,
#         batch=batch,
#         pos=pos,
#         label=label,
#     ))
token_df = nutils.make_token_df(big_tokens, len_prefix=8, len_suffix=3)
layer = 31
ni = 892
label = f"L{layer}N{ni}"
token_df[label] = to_numpy(big_cache["post", layer][:, :, ni]).flatten()
nutils.focus_df_column(token_df, label)
# %%
layer = 31
ni = 892
s = "For every action there is"
temp_logits, temp_cache = model.run_with_cache(s)
print("an vs a logit diff", temp_logits[:, -1, AN] - temp_logits[:, -1, A])
print("Neuron act", temp_cache["post", layer][0, -1, ni])

win = model.W_in[layer, :, ni]
temp_resid_stack, temp_labels = temp_cache.get_full_resid_decomposition(layer, True, True, True, -1, True)
print("Total sum of attribution, to sanity check", (temp_resid_stack @ win).sum())
temp_df = (pd.DataFrame(dict(x=to_numpy(temp_resid_stack.squeeze(1) @ win)), index=temp_labels).sort_values("x", ascending=False).head(20))
temp_df["frac"] = temp_df["x"]  / temp_cache["post", layer][0, -1, ni].item()
nutils.show_df(temp_df)
line(temp_resid_stack @ win, x=temp_labels, title=f"Linear Attribution on neuron L{layer}N{ni}")
# %%
layer = 26
head = 1
pattern = big_cache["pattern", layer][:, head, :, :]
max_pattern, argmax_pattern = pattern[:, :, 1:].max(-1)
argmax_pattern+=1
argmax_src_tokens = big_tokens[einops.repeat(torch.arange(len(big_tokens)), "x -> x p", p=big_tokens.shape[-1]).cuda(), argmax_pattern]
argmax_src_str_tokens = [nutils.process_tokens(argmax_src_tok) for argmax_src_tok in argmax_src_tokens]
token_df = nutils.make_token_df(big_tokens, len_prefix=8, len_suffix=3)
token_df["src_tok"] = nutils.list_flatten(argmax_src_str_tokens)
token_df["max_pattern"] = to_numpy(max_pattern).flatten()
token_df["argmax_pattern"] = to_numpy(argmax_pattern).flatten()
token_df["bos_pattern"] = to_numpy(pattern[:, :, 0]).flatten()
nutils.focus_df_column(token_df, "max_pattern")
# %%
px.histogram(token_df, x="max_pattern", marginal="box", hover_name="context", title="Max non-BOS Attention for head L26H1").show()
px.histogram(token_df, x="bos_pattern", marginal="box", hover_name="context", title="BOS Attention for head L26H1").show()
# %%
s = "I climbed up the pear tree and picked a pear. I climbed up the apricot tree and picked"
temp_logits, temp_cache = model.run_with_cache(s)
token_list = nutils.process_tokens_index(s)
# imshow(temp_cache["pattern", 26][0, 1, :, :], x=token_list, y=token_list, title="Attention for head L26H1")
print(f"Neuron Act: {temp_cache['post', 31][0, -1, 892].item():.3f}")
print(f"Attn: {' '*6}{temp_cache['pattern', 26][0, 1, -1, 16].item():.3f}")
print(f"Logit Diff: {temp_logits[0, -1, AN].item() - temp_logits[0, -1, A].item():.3f}")
# %%
common_words = open("common_words.txt", "r").read().split("\n")
print(common_words[:10])

num_tokens = [len(model.to_tokens(" "+word, prepend_bos=False).squeeze(0)) for word in common_words]
print(list(zip(num_tokens, common_words))[:10])

word_df = pd.DataFrame({"word": common_words, "num_tokens": num_tokens})
word_df = word_df.query('num_tokens < 4')
word_df.value_counts("num_tokens")

# %%
# num_words = 300
# word_list = word_df.query("num_tokens==1").sample(300).word.to_list()
common_objects = [
    "pen", "hat", "cup", "bag", "box", "car", "dog", "cat", "key", "bed",
    "pot", "pan", "jar", "jug", "rug", "bat", "ball", "shoe", "ship", "bike",
    "desk", "door", "bell", "book", "bowl", "coin", "comb", "cord", "disk", "doll",
    "drum", "flag", "fork", "lamp", "lock", "mug", "nail", "pipe", "ring", "rope",
    "seed", "skirt", "spoon", "stamp", "star", "stick", "tent", "tie", "tooth", "toy",
    "tree", "watch", "whip", "bird", "boat", "boot", "cane", "card", "chain", "chair",
    "chalk", "clock", "cloth", "cloud", "coat", "crab", "disk", "dress", "drop", "drum",
    "duck", "dust", "fence", "flag", "floor", "flower", "fly", "fog", "fork", "fruit",
    "glass", "glove", "grass", "hair", "hand", "harp", "hat", "hill", "horn", "horse",
    "house", "island", "jewel", "jug", "kettle", "key", "kite", "knife", "leaf", "leg",
    "library", "light", "line", "loaf", "lock", "machine", "man", "map", "moon", "net",
    "nose", "nut", "office", "orange", "oven", "parcel", "pen", "pencil", "picture", "pig",
    "pin", "pipe", "plane", "plate", "plough", "pocket", "pot", "potato", "prison", "pump",
    "rail", "rat", "receipt", "ring", "rod", "roof", "root", "sail", "school", "scissors",
    "screw", "seed", "sheep", "shelf", "ship", "shirt", "shoe", "skin", "skirt", "snake",
    "sock", "spade", "sponge", "spoon", "spring", "square", "stamp", "star", "station", "stem",
    "stick", "stocking", "stomach", "store", "street", "sun", "table", "tail", "thread", "throat",
    "thumb", "ticket", "toe", "tongue", "tooth", "town", "train", "tray", "tree", "trousers",
    "umbrella", "wall", "watch", "wheel", "whistle", "window", "wire", "wing", "worm", "yarn"
]
vowel_objects = [
    "apple", "apron", "arm", "ankle", "arrow", "atom", "ant", "anchor", "album", "axe",
    "ear", "egg", "elbow", "engine", "eagle", "earring", "envelope", "eye", "eel", "earth",
    "ice", "iron", "ink", "island", "ivy", "igloo", "insect", "instrument", "image", "indicator",
    "oak", "oar", "ocean", "octopus", "onion", "orange", "organ", "oven", "owl", "ox",
    "umbrella", "urn", "utensil", "uniform", "ukelele", "unit", "unicorn", "upstairs", "underwear", "urchin",
    "emerald", "end", "elephant", "elm", "easel", "eraser", "eskimo", "entrance", "estate", "echo",
    "ash", "art", "armchair", "air", "arch", "anvil", "alloy", "alley", "atom", "amulet",
    "olive", "opera", "opal", "ottoman", "orchid", "orbit", "ostrich", "oxen", "oil", "ounce",
    "iceberg", "iris", "idea", "iguanodon", "inlet", "icon", "input", "isle", "itch", "issue",
    "udder", "uplift", "update", "upgrade", "undo", "uptake", "upbeat", "upturn", "upload", "upstream",
    "antenna", "almond", "arena", "aorta", "ape", "asteroid", "aster", "auction", "audio", "avocado",
    "edge", "eel", "eel", "equipment", "escalator", "essence", "emblem", "echo", "engineer", "equator",
    "opal", "orchard", "oboe", "oval", "oven", "overcoat", "oyster", "ounce", "outlet", "outline",
    "aerial", "airplane", "awning", "award", "agent", "agate", "arc", "arena", "armadillo", "apricot"
]
word_list = common_objects[:50] + vowel_objects[:50]
token_lengths = [len(model.to_tokens(" "+word, prepend_bos=False).squeeze(0)) for word in word_list]
word_list = [word_list[i] for i in range(len(word_list)) if token_lengths[i]==1]
len(word_list)
# word_list
# %%
prompt_template = "I climbed up the pear tree and picked a pear. I climbed up the {} tree and picked"
prompt_list = [prompt_template.format(word) for word in word_list]
tree_tokens = model.to_tokens(prompt_list)
model.to_str_tokens(tree_tokens[0])
# %%
tree_logits, tree_cache = model.run_with_cache(tree_tokens, names_filter=lambda name: "pattern" in name or "post" in name or "z" in name or "out" in name or "scale" in name)
tree_neuron_acts = tree_cache['post', 31][:, -1, 892]
tree_attn = tree_cache['pattern', 26][:, 1, -1, 16]
tree_logit_diff = tree_logits[:, -1, AN] - tree_logits[:, -1, A]
tree_df = pd.DataFrame(dict(
    word=word_list,
    neuron_acts=to_numpy(tree_neuron_acts),
    attn=to_numpy(tree_attn),
    logit_diff=to_numpy(tree_logit_diff),
    is_vowel = [word[0].lower() in "aeiou" for word in word_list],
))
nutils.focus_df_column(tree_df, "logit_diff")
# %%
px.histogram(tree_df, x="logit_diff", color="is_vowel", marginal="rug", hover_name="word", title="Logit Diff for Vowel vs Consonant Words", barmode="overlay", histnorm='percent').show()
px.histogram(tree_df, x="neuron_acts", color="is_vowel", marginal="rug", hover_name="word", title="Neuron Act L31N892 for Vowel vs Consonant Words", barmode="overlay", histnorm='percent').show()
px.histogram(tree_df, x="attn", color="is_vowel", marginal="rug", hover_name="word", title="Attn L26H1 for Vowel vs Consonant Words", barmode="overlay", histnorm='percent').show()
# %%
# tree_resid_stack, tree_labels = tree_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
unembed_dir = model.W_U[:, AN] - model.W_U[:, A]
tree_dla = []
tree_labels = model.all_head_labels() + [f"MLP{i}" for i in range(n_layers)]
tree_z = tree_cache.stack_activation("z")[:, :, -1, :, :]
W_O = model.W_O
tree_z_dla = (einops.einsum(tree_z, W_O, unembed_dir, "layer batch head d_head, layer head d_head d_model, d_model -> layer head batch"))
tree_dla.append(einops.rearrange(tree_z_dla, "layer head batch -> batch (layer head)"))
tree_mlp_out = tree_cache.stack_activation("mlp_out")[:, :, -1, :]
tree_dla.append((tree_mlp_out @ unembed_dir).T)
tree_dla = torch.cat(tree_dla, dim=-1)
all_dla = tree_dla.mean(0)
vowel_dla = (tree_dla[tree_df.is_vowel]).mean(0)
consonant_dla = (tree_dla[~tree_df.is_vowel]).mean(0)
diff_dla = vowel_dla - consonant_dla
line([all_dla, vowel_dla, consonant_dla, diff_dla], x=tree_labels, title="Linear Attribution on Logit Diff", line_labels=["all", "vowel", "consonant", "diff"])
# %%

mlp_layers = [23, 29, 31, 34, 35]
for layer in mlp_layers:
    acts = tree_cache["post", layer][:, -1, :]
    W_out = model.W_out[layer]
    wdla = W_out @ unembed_dir
    dla = acts * wdla[None, :]
    all_dla = dla.mean(0)
    vowel_dla = (dla[tree_df.is_vowel]).mean(0)
    consonant_dla = (dla[~tree_df.is_vowel]).mean(0)
    diff_dla = vowel_dla - consonant_dla
    line([all_dla, vowel_dla, consonant_dla, diff_dla], title=f"Linear Attribution on Logit Diff for MLP{layer}", line_labels=["all", "vowel", "consonant", "diff"])


# %%
def get_dla(vec, layer=None, mlp_input=False, apply_ln=True):
    # unembed_dir = model.W_U[:, AN] - model.W_U[:, A]
    if layer is None:
        layer = n_layers
    if mlp_input:
        mlp_layer = layer + 1
    else:
        mlp_layer = layer

    tree_dla = []
    tree_labels = [f"L{l}H{h}" for l in range(layer) for h in range(n_heads)] + [f"MLP{i}" for i in range(mlp_layer)]
    tree_z = tree_cache.stack_activation("z")[:, :, -1, :, :]
    W_O = model.W_O
    tree_z_dla = (einops.einsum(tree_z, W_O, unembed_dir, "layer batch head d_head, layer head d_head d_model, d_model -> layer head batch"))
    tree_dla.append(einops.rearrange(tree_z_dla, "layer head batch -> batch (layer head)"))
    tree_mlp_out = tree_cache.stack_activation("mlp_out")[:, :, -1, :]
    tree_dla.append((tree_mlp_out @ unembed_dir).T)

# %%
vec = model.W_in[31, :, 892]
layer = 23
acts = tree_cache["post", layer][:, -1, :]
W_out = model.W_out[layer]
wdla = W_out @ vec
dla = acts * wdla[None, :]
all_dla = dla.mean(0)
vowel_dla = (dla[tree_df.is_vowel]).mean(0)
consonant_dla = (dla[~tree_df.is_vowel]).mean(0)
diff_dla = vowel_dla - consonant_dla
line([all_dla, vowel_dla, consonant_dla, diff_dla], title=f"Linear Attribution on Logit Diff for MLP{layer}", line_labels=["all", "vowel", "consonant", "diff"])

# %%
vec = unembed_dir
layer = None
mlp_input = False
pos = -1
def get_dla(tree_cache, vec, layer=None, mlp_input=False, pos=-1, apply_ln=True):
    if layer is None:
        layer = n_layers
    if mlp_input:
        attn_layer = layer + 1
    else:
        attn_layer = layer
    mlp_layer = layer
    
    if layer==n_layers:
        ln_scale = tree_cache["scale"][:, pos, 0]
    elif mlp_input:
        ln_scale = tree_cache["scale", layer, "ln2"][:, pos, 0]
    else:
        ln_scale = tree_cache["scale", layer, "ln1"][:, pos, 0]


    tree_dla = []
    tree_labels = [f"L{l}H{h}" for l in range(attn_layer) for h in range(n_heads)] + [f"L{l}N{n}" for l in range(mlp_layer) for n in range(d_mlp)]
    tree_z = tree_cache.stack_activation("z")[:attn_layer, :, pos, :, :]
    W_O = model.W_O[:attn_layer]
    tree_z_dla = (einops.einsum(tree_z, W_O, unembed_dir, "layer batch head d_head, layer head d_head d_model, d_model -> layer head batch"))
    tree_dla.append(einops.rearrange(tree_z_dla, "layer head batch -> batch (layer head)"))

    tree_neuron_acts = tree_cache.stack_activation("post")[:mlp_layer, :, pos, :]
    W_out = model.W_out[:mlp_layer]
    W_out_vec = W_out @ vec
    wdla = tree_neuron_acts * W_out_vec[:, None, :]
    wdla = einops.rearrange(wdla, "layer batch neuron -> batch (layer neuron)")
    tree_dla.append(wdla)
    tree_dla = torch.cat(tree_dla, dim=-1)
    tree_dla = tree_dla / ln_scale[:, None]

    all_dla = tree_dla.mean(0)
    vowel_dla = (tree_dla[tree_df.is_vowel]).mean(0)
    consonant_dla = (tree_dla[~tree_df.is_vowel]).mean(0)
    diff_dla = vowel_dla - consonant_dla
    return torch.stack([all_dla, vowel_dla, consonant_dla, diff_dla]), tree_labels
LINE_LABELS = ["all", "vowel", "consonant", "diff"]
stack, labels = get_dla(tree_cache, vec, layer, mlp_input, pos)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
# %%
stack, labels = get_dla(tree_cache, model.W_in[31, :, 892], 31, True, -1)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
# %%
tree_cache["post", 31][:, -1, 892][~tree_df.is_vowel].mean()
# %%
stack, labels = get_dla(tree_cache, model.W_V[26, 1] @ model.W_O[26, 1] @ model.W_in[31, :, 892], 26, False, 16)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
# %%
stack, labels = get_dla(tree_cache, model.W_in[15, :, 4007], 15, True, 16)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
stack, labels = get_dla(tree_cache, model.W_in[0, :, 1595], 0, True, 16)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
stack, labels = get_dla(tree_cache, model.W_in[7, :, 2183], 7, True, 16)
# line(stack, x=labels, line_labels=LINE_LABELS)
temp_df = pd.DataFrame(dict(all=to_numpy(stack[0]), vowel=to_numpy(stack[1]), consonant=to_numpy(stack[2]), diff=to_numpy(stack[3])), index=labels)
nutils.focus_df_column(temp_df, "diff")
# %%
s = " ".join(word_list)
temp_logits, temp_cache = model.run_with_cache(s)
nutils.create_html(model.to_str_tokens(s), temp_cache["post", 16][0, :, 4543])
# %%
consonant_z_ave = tree_cache.stack_activation("z")[:, ~tree_df.is_vowel].mean(1)
vowel_z_ave = tree_cache.stack_activation("z")[:, tree_df.is_vowel].mean(1)
def mean_ablate_z(z, hook, layer, head):
    z[:, :, head, :] = consonant_z_ave[layer, :, head][None]
    return z
def mean_denoise_z(z, hook, layer, head):
    z[:, :, head, :] = vowel_z_ave[layer, :, head][None]
    return z
attn_ablate_hooks = [
    (utils.get_act_name("z", 26), partial(mean_ablate_z, layer=26, head=1))
]
attn_denoise_hooks = [
    (utils.get_act_name("z", 26), partial(mean_denoise_z, layer=26, head=1))
]

consonant_neuron_ave = tree_cache.stack_activation("post")[:, ~tree_df.is_vowel].mean(1)
vowel_neuron_ave = tree_cache.stack_activation("post")[:, tree_df.is_vowel].mean(1)
def mean_ablate_neuron(act, hook, layer, ni, pos):
    act[:, pos, ni] = consonant_neuron_ave[layer, pos, ni][None]
    return act
def mean_denoise_neuron(act, hook, layer, ni, pos):
    act[:, pos, ni] = vowel_neuron_ave[layer, pos, ni][None]
    return act
neuron_ablate_hooks = [
    (utils.get_act_name("post", 31), partial(mean_ablate_neuron, layer=31, ni=892, pos=-1)),
    (utils.get_act_name("post", 35), partial(mean_ablate_neuron, layer=35, ni=1884, pos=-1)),
]
neuron_denoise_hooks = [
    (utils.get_act_name("post", 31), partial(mean_denoise_neuron, layer=31, ni=892, pos=-1)),
    (utils.get_act_name("post", 35), partial(mean_denoise_neuron, layer=35, ni=1884, pos=-1)),
]
neuron_ablate_mid_hooks = [
    (utils.get_act_name("post", 0), partial(mean_ablate_neuron, layer=0, ni=1595, pos=16)),
    (utils.get_act_name("post", 7), partial(mean_ablate_neuron, layer=7, ni=2183, pos=16)),
    (utils.get_act_name("post", 15), partial(mean_ablate_neuron, layer=15, ni=4007, pos=16)),
]
neuron_denoise_mid_hooks = [
    (utils.get_act_name("post", 0), partial(mean_denoise_neuron, layer=0, ni=1595, pos=16)),
    (utils.get_act_name("post", 7), partial(mean_denoise_neuron, layer=7, ni=2183, pos=16)),
    (utils.get_act_name("post", 15), partial(mean_denoise_neuron, layer=15, ni=4007, pos=16)),
]

def normalise_logit_diff(diff):
    tree_logit_diff = tree_logits[:, -1, AN] - tree_logits[:, -1, A]
    return (diff - tree_logit_diff[~tree_df.is_vowel].mean(0))/(tree_logit_diff[tree_df.is_vowel].mean(0) - tree_logit_diff[~tree_df.is_vowel].mean(0))

for hooks, label in [
    (attn_ablate_hooks, "attn_ablate_hooks"),
    (attn_denoise_hooks, "attn_denoise_hooks"),
    (neuron_ablate_hooks, "neuron_ablate_hooks"),
    (neuron_denoise_hooks, "neuron_denoise_hooks"),
    (neuron_ablate_mid_hooks, "neuron_ablate_mid_hooks"),
    (neuron_denoise_mid_hooks, "neuron_denoise_mid_hooks"),
    (attn_ablate_hooks + neuron_ablate_hooks + neuron_ablate_mid_hooks, "all_ablate_hooks"),
    (attn_denoise_hooks + neuron_denoise_hooks + neuron_denoise_mid_hooks, "all_denoise_hooks"),
]:
    patched_tree_logits = model.run_with_hooks(tree_tokens, fwd_hooks = hooks)
    patched_tree_logit_diff = patched_tree_logits[:, -1, AN] - patched_tree_logits[:, -1, A]
    
    print(label)
    print(f"Vowel logit diff: {normalise_logit_diff(patched_tree_logit_diff[tree_df.is_vowel].mean()).item():.2%}")
    print(f"Consonant logit diff: {normalise_logit_diff(patched_tree_logit_diff[~tree_df.is_vowel].mean()).item():.2%}")
    print()

# patched_tree_logits = model.run_with_hooks(tree_tokens, fwd_hooks = neuron_denoise_hooks)
# patched_tree_logit_diff = patched_tree_logits[:, -1, AN] - patched_tree_logits[:, -1, A]
# print(f"Vowel logit diff: {patched_tree_logit_diff[tree_df.is_vowel].mean().item():.3f}")
# print(f"Consonant logit diff: {patched_tree_logit_diff[~tree_df.is_vowel].mean().item():.3f}")

# %%
