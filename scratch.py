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
def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]
    
    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))
token_df = make_token_df(big_tokens, len_prefix=8, len_suffix=3)
layer = 31
ni = 892
label = f"L{layer}N{ni}"
token_df[label] = to_numpy(big_cache["post", layer][:, :, ni]).flatten()
nutils.focus_df_column(token_df, label)
# %%
