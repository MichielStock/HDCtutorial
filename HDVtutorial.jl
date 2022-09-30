### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 366fc4c8-2d15-11ed-18cd-c14f15c96f6e
using LinearAlgebra, Plots, StatsBase, SparseArrays, Colors, Random, PlutoUI

# ╔═╡ 3cdc0adf-2ba6-4213-b644-907533de412f
using DataFrames, CSV

# ╔═╡ 7f6763c0-c440-4e9d-a514-98ec4273bd57
begin
	using FASTX
	
	io = open("data/sapiens_yeast_proteins.fasta", "r")
	reads = [(description(rec), sequence(rec)) for rec in FASTAReader(io)]

	headers = first.(reads)
	sequences = last.(reads)

	n_seqs = length(sequences)

	species = [occursin("YEAST", h) ? "YEAST" : "HUMAN" for h in headers]

	humane = species .== "HUMAN"
end;

# ╔═╡ f6b4fc92-51fa-42ba-8f3e-6f81bc6d8624
md"""
# Tutorial on Hyperdimensional Computing

**Michiel Stock**

September 2022
"""

# ╔═╡ a3820f10-9930-4dba-85aa-80a09744578d
md"""
## Building a brain for dummies

Artificial neural networks, despite the name, are still very different from how brains work. Most networks are trained using top-down gradient-based methods that are not remotely biologically realistic. Training a state-of-the-art deep learning model such as GTP-3 or DALL-E2 can cost millions of dollars and requires specific infrastructure to deploy. Meanwhile, our brains generate general intelligence, trained in an unsupervised way. The brain has an energy consumption of about 20 Watt, which is about three cans of coke a day. Brains are remarkably efficient.

Hyperdimensional computing (HDC) is a relatively new approach to artificial intelligence. It tries to mimic some aspects of brain functioning more closely. Here, we compute with hyperdimensional vectors (HDV), which is merely a fancy name for vectors of huge dimensionality, typically 10,000. Each HDV can represent a concept, for example, a word in a language, a position along an axis, an ingredient in a recipe or an atom in a molecule. Using a set of basic, well-chosen operations, one can combine these atomic vectors into new vectors: from words to sentences, from coordinates to a position in space, from ingredients to a recipe. These rules allow the user to create *structured* and *hierarchical* representations of whatever they want to model. 

Hyperdimensional computing exploits the power of high dimensions by smearing out the information over the complete vector. A distributed, holographic representation allows building a remarkably robust and efficient learning system. Because HDC can be computed using efficient bit-operations, it can be made very energy-efficient. Furthermore, a basic HDC system is relatively easy to implement, as we will see. 
"""

# ╔═╡ d3f992f2-bd1e-4544-809f-b9fab1c11005
TableOfContents()

# ╔═╡ 59f5ad60-4554-462e-9a7b-1b7e465b720b
md"""## Building blocks of HDC

To build a HDC system we need the following building blocks:
1) HDVs themselves. These HDVs are just ordinary vectors. 
2) Suited arithmetic operations to manipulate these vectors.
3) Metrics to compute similarities between HDVs.
"""

# ╔═╡ 09efbacf-8ee9-46c1-a4a7-11921fbc783b
const N = 10_000

# ╔═╡ 0424b37b-64f6-48c6-8e61-85b6e62fc93c
md"""
### Generating hyperdimensional vectors

Hyperdimensional vectors are nothing more than vectors of a very high dimension. The dimensionality should be sufficiently large that one can comfortably store all the concepts of the system of interest. For our system to be robust, the space of vectors should also be large enough that *a randomly-drawn vector is almost surely unrelated to meaningful vectors*. In this notebook, we work with a dimensionality of $N =$ $N.
"""

# ╔═╡ b5358dae-d136-4343-a3d9-d58ab5f1a79c
md"""
The nature of the vectors, whether filled with real, binary or bipolar (i.e. -1 and 1's), is less critical than the vectors being *large*. Of course, the vectors' nature will determine the associated operators' choice. 

Typically, one uses either binary vectors or bipolar vectors. The former can use highly efficient bit operations, and the latter has elementary mathematics. We will choose the latter. To ease the explanation, I will still refer to the individual elements of such vectors as 'bits'.

We assign a randomly-chosen HDV to atomic concepts. Generating a random HDV is easy.
"""

# ╔═╡ 0418d5fd-df04-4d56-ba7c-d45d18aa161d
hdv() = rand((-1,1), N)

# ╔═╡ 49018537-b529-4d07-b1cb-d34fe58294f0
md"We expect about half of the elements of two randomly-chosen vectors to match."

# ╔═╡ 0c72733c-897b-4d9b-84f7-98efe2f197de
md"""
Our two randomly generated vectors show that about half of the $N$ elements match. The expected value is $N/2$, with a variance ($\sigma^2$) of $N/4$ and thus the standard deviation ($\sigma$) is $\sqrt{N}/2$. 

> Any two randomly chosen HDV likely share between $(N-3\sqrt{N})$ and $(N+3\sqrt{N})$ elements.
"""

# ╔═╡ f34d034a-942b-41a0-92ec-23608a43f778
md"""
When we need a set of `n` vectors, it makes sense to generate a matrix in one go.
"""

# ╔═╡ 4c647ad6-3e70-48f6-b778-8f9a34d58a6f
hdv(n) = rand((-1,1), n, N)

# ╔═╡ 67dc0274-ae49-4262-ab9d-a7716ab7e851
x = hdv()  # a HDV!

# ╔═╡ 8123aaec-aa78-495e-83a1-3a4f4b244000
y = hdv()  # another one!

# ╔═╡ ea146895-6d82-4dcb-a24d-0dae70ae39b9
sum(x.==y)

# ╔═╡ 7ea554a1-5efb-461f-9654-2abff0dc80b5
V = hdv(10)  # each row is a HDV

# ╔═╡ 1a7bbd06-b6d8-4e0a-a6d7-cb082b9bb69d
md"""
### Bundling

Our first operation is **bundling** or aggregation. This operation combines two or more HDVs in a new HDV that *is similar to all elements in the set*. For bipolar vectors, the element-wise majority fits the bill. Note that when we bundle two HDVs, many of the elements will be set to zero, indicating that the corresponding elements of the parents were in disagreement. 
"""

# ╔═╡ a40b8155-34f6-47cc-aaa0-82069ca2dfb5
bundle(U::Matrix{Int}) = sum(U, dims=1)[:] .|> sign

# ╔═╡ eebcab6e-a7e9-4cfd-9e6e-3e39c5d4b8f6
bundle(xs::Vector{Int}...) = reduce(.+, xs) .|> sign

# ╔═╡ 119cd151-26b8-4d4e-a919-88c54cac8616
bundle(x, y) 

# ╔═╡ adf79e3a-26f3-4afe-bd14-fc48880a29ec
bundle(V)  # bundling makes most sense to find agreement among several HDVs

# ╔═╡ 3578bb7d-353d-4f48-9f50-50b4aaaad19d
md"""
### Binding

The second operator is **binding**: combining two vectors in a new vector different from both. Bundling encodes an *interaction* between two concepts. For bipolar vectors, element-wise multiplication has the desired properties. When two elements are in agreement ((-1, -1) or (1, 1)), the result will be 1. When they are in disagreement ((-1, 1) or (-1, 1)), the result will be -1. One typically uses the XOR function for binary vectors, generating a `true` only when the input bits differ.
"""

# ╔═╡ f8170a5b-4c19-4c64-ad3a-ab82a8b448c6
bind(xs::Vector{Int}...) = reduce(.*, xs)

# ╔═╡ 591e5938-3bba-4b72-8d20-0848884e732b
bind(x, y)

# ╔═╡ 3b543c80-8f36-49e9-a2fc-a80b6677ad1e
md"Binding is reversible:"

# ╔═╡ 9d2ceb8b-61ea-49c5-b3a4-08b061423e7c
bind(bind(x, y), y) == x

# ╔═╡ 66440162-3b60-4078-bdcf-3426403fcc2f
md"""
### Shifting

Our last operation to create new vectors is a simple unitary one that can create a new HDV from an old one by shifting its elements. For example, this is important when one wants to include positional information (e.g., in a sentence). An easy trick is to perform **shifting** using cyclic permutation. Just move every element of the HDV one or more steps to the right. Elements that fall off are placed back at the beginning. 
"""

# ╔═╡ 4e956ea8-fd2d-4fdb-be33-f8987b2bd1e5
shift(x, k=1) = circshift(x, k)

# ╔═╡ b2dc53a5-0bcd-4551-bf10-9ce405e8b651
shift(x)

# ╔═╡ abab4185-4b15-4457-aa63-3c825477ae48
sum(x.==shift(x))  # shifted vector only shares half its elements with the old one

# ╔═╡ a1eedfc9-4ee4-4f0c-8f89-1fe70b71a686
md"""
### Similarity between HDVs

Finally, we need a way to quantify the **similarity** between two HDVs. This will allow us to detect related patterns. A simple way would be to count the number of matching bits, which we have shown to be expected around $N/2$ for two randomly drawn HDVs. 

For bipolar HDV, the **cosine similarity** is a good similarity measure:

$$\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}||\, ||\mathbf{y}||}\,.$$

The cosine similarity always lies between -1 and 1.
"""

# ╔═╡ 8b153389-1da3-47b4-b747-ec252e08d90d
cos(x, y) = dot(x, y) / (norm(x) * norm(y))

# ╔═╡ 59425927-abaa-45a9-a5e8-1619ae806840
cos(x, y)  # random vectors, so close to 0

# ╔═╡ c5a8dd2f-b134-4b7b-96f3-3ad3611d92a4
md"""
### Properties of the operations

Bundling, binding, and shifting allow for meaningful computation on the concepts your HDVs represent. One vital property of binding and shifting possess is that they are *reversible*: applying them does not remove any information. Binding also distributes over bundling:
"""

# ╔═╡ 3bd2b8c8-7a57-4080-8861-f231dfa4471e
z = hdv()

# ╔═╡ 3ce916d7-935d-43ee-b5dc-10651f281985
bind(x, bundle(y, z)) == bundle(bind(x, y), bind(x, z))

# ╔═╡ efab3c5b-bb21-486e-8474-6b8440c9e42b
md"This means that transforming a bundle of concepts with binding is equivalent to binding every element before bundling.

Binding also preserves distance and similarity."

# ╔═╡ a7d1326e-250c-4fd7-af2b-2f5e0504f03e
cos(x, y) ≈ cos(bind(x, z), bind(y, z))

# ╔═╡ 8fc6f3a0-e05d-44cc-b3e3-bf35c1d02918
md"Similarly, permutation or shifting also perserves the similarity between two vectors."

# ╔═╡ 432a9fae-1c68-4312-8837-9fe489e26791
cos(x, y) ≈ cos(shift(x), shift(y))

# ╔═╡ b5e16410-82eb-4851-a20d-92106c2654e9
md"""
## The blessing of high dimensions

While some machine learning methods reduce dimensionality, the magic of HDC arises by *expanding* the dimensionality. Feature expansion is far from a novel concept: deep learning does it, kernel methods do it etc. Moving your problem to vast dimensionality can make it easier to solve. You could comfortably fit, for example, all of organic chemistry in such a system. The vast majority of the vectors in such a space are uncorrelated, as shown below.
"""

# ╔═╡ 001f44f6-7e3f-4312-8387-9587f7811fdb
let
	p = histogram([cos(hdv(), hdv()) for i in 1:1000], xlims=(-1,1), label="", xlabel="cos. sim.", title="Cosine similarity between randomly chosen HDVs")
	vline!(p, [0], label="mean", lw=2)
end

# ╔═╡ 84a318f7-3d7c-4809-91ce-d3388f915b3f
md"""
As you can see, most HDVs are approximately orthogonal to one other. This implies  that when two vectors *are* related (share substantially more than 50% of their bits), this is a good indication that they are related semantically! To put a number on this, there is less than a one per cent chance that two random HDVs share more dan 5500 bits, corresponding to a cosine similarity of 0.05. This makes HDC very sensitive to picking up related concepts.

Every vector also has billions of neighbours that differ only a few bits. So, x and y may be both related to z, while x and y themselves are unrelated. This property allows one to build semantic links in the space. Bird and sock might be unrelated, but they can be conceptually linked as bird ~ chicken ~ drumstick ~ leg ~ foot ~ sock.
"""

# ╔═╡ 27e6169f-1bfa-4e59-bf0f-872a432983d5
md"""
## Examples

Let us explore some didactic examples to show how objects, numbers, colours, and sequences can be represented as HDVs and how we can learn with them.

"""

# ╔═╡ c6718aac-f7e9-47b7-9698-d52c31208db9
md"""
### Example 1: colour matching

Let us start with a simple toy example. We have a list of emojis, each with associated colours. Can we find the *average* colour per emoji? 

Take a look at the data.
"""

# ╔═╡ 853a5ec0-4a9a-4430-b0f2-b3e7994c25ba
md"""
Colors are a bit more tricky. This is an example color:
"""

# ╔═╡ a193a721-3caa-4069-b53f-3cdb58cb075e
md"""
We see that a colour can be represented by three numbers: the fractions of red, green, and blue. Every value is just a number between 0 and 1. If we can construct an embedding for numbers, we can represent a colour as a *binding* of three numbers.

Representing numbers in a fixed interval $[a, b]$ with HDVs is relatively easy. We first divide the interval into $k$ equal parts. Then, we generate an HDV representing the lower bound of the interval. We replace a fraction of $1/k$ of the previous vector for every step with fresh random bits.
"""

# ╔═╡ ec7cfa4e-d394-4b2a-9a4a-160c080fa341
function range_hdvs(steps)
	k = length(steps) - 1
	V = hdv(k+1)
	for i in 2:k+1
		for j in 1:N
			V[i,j] = rand() < 1 / k ? -V[i-1,j] : V[i-1,j]
		end
	end
	return V
end

# ╔═╡ 16e9a27f-0275-48b3-bbc3-9394725ec595
md"Let us represent a color chanel from 0 to 1 in steps of 0.05. This resolution should suffice." 

# ╔═╡ c6ecd79c-0285-47a2-9c33-af440f5d3325
color_steps = 0:0.05:1

# ╔═╡ 0a1227c0-6557-46e7-8dcb-e7867b1aac94
const reds_hdv = range_hdvs(color_steps)

# ╔═╡ 0f174703-27df-4c07-a03d-95cf482a1c1d
md"Take a look at the correlation. HDVs that represent numbers closer to onother are more similar:"

# ╔═╡ 7976d118-a1e8-43d8-862b-a21a05a7306e
let
	k = length(color_steps)
	S = [cos(reds_hdv[i,:], reds_hdv[j,:]) for i in 1:k, j in 1:k]
	heatmap(color_steps, color_steps, S, title="cosine similarity between range HDVs")
end

# ╔═╡ 6aa8efb1-8705-474d-bd79-a75d78643f6c
md"Repeat for green and blue. It is important we have fresh HDVs so that our channels are orthogonal!"

# ╔═╡ 9c2b8b28-751d-4def-ae1c-9b1bf545f23f
const greens_hdv = range_hdvs(color_steps);

# ╔═╡ d751b589-8c7d-48d5-9548-3fe105b969e8
const blues_hdv = range_hdvs(color_steps);

# ╔═╡ 909a155e-5cd0-4360-ac9c-4761521293db
md"""
A colour is just a triple of the RGB channel (other representations such as hue-value-saturation might be better, but RGB works fine for our purposes). We can create a colour HDV by binding the three channel-HDVs.
"""

# ╔═╡ 51e5be0b-4dfa-40c3-bbda-bd3b44595c86
col_ind(v) = round(Int, v * 20) + 1

# ╔═╡ 5eb33b1a-f91e-481c-893b-7e179a8136e5
encode_col(col) = bind(reds_hdv[col_ind(col.r),:], greens_hdv[col_ind(col.g),:], blues_hdv[col_ind(col.b),:])

# ╔═╡ 3de9f13e-4a6f-4ba0-b2d4-86fa617c7317
md"Try it on our example colour:"

# ╔═╡ aab48f4b-10da-4872-aee2-bec419f867d2
md"""
Mapping from colour to HDV is straightforward. The big question is, can we do the reverse: *going from HDV to the corresponding colour*? This is an inverse problem and is generally difficult to solve. We will deal with the issue by using a simple *table lookup*:

1. generate a list of many random colours;
2. get the HDV for each colour;
3. to map a new HDV $\mathbf{x}$ to a colour, just look for the most similar colour in the database using the cosine similarity and return it.
"""

# ╔═╡ 3f51851d-bfbd-4301-880f-923851305bf5
md"Now that we can encode emoji and colours, we can represents couples with yet another binding operation. Subsequently, we can embed the complete dataset by bundling."

# ╔═╡ 93c47ae3-1fcf-470f-90c6-28dad06631ad
md"From the dataset embedding, we can now extract all the 'average' colours for each emoji. Remember, by binding again with the fire truck, we reverse the initial binding."

# ╔═╡ 8e334f45-08d0-455b-9a99-a1978cb9ae61
md"""
This seems to work quite well! Let us move to a variation to this problem. With every emoji, we now give *three* colours: one is related to the emoji (we don't know which one), and two are randomly generated. Can we still find the matching colour?
"""

# ╔═╡ 4ede9c89-a64a-40f8-898a-22b7edb7d616
md"""
This is an example of a *multi-instance problem*: for every observation, we have several labels, and we know at least one is relevant. It pops up frequently in weakly-supervised learning, for example, when you have an image containing multiple objects with only a label for the image or when a molecule has different configurations, knowing that at least one is chemically active.

For HDC, dealing with multi-instance learning is a breeze. We can bundle the different colours. Due to distributivity, the irrelevant colours will act as noise and will be averaged away when bundling the complete data.
"""

# ╔═╡ 02f703ff-f44f-445d-bfb6-9aa3273c0393
md"Not bad right? Dealing with this issue is quite complex in most machine learning algorithms but HDC gives us the tools to model the structure of our problem."

# ╔═╡ 00b949c3-5986-4f72-a1db-45f5fa320d91
md"""
### Example 2: recipes

Let us move to a real dataset, the recipes dataset of [Ahn et al.](https://www.nature.com/articles/srep00196). It can learn us which regions tend to use which ingredients. Here, there are over 55,000 recipes from all over the world that are represented as bags of ingredients.
"""

# ╔═╡ 2df88a19-6965-4e48-92f9-3725ff86f7ad
md"These are all the ingredients:"

# ╔═╡ 6f8d9c88-1686-4a54-9cf6-ce53c31629a8
md"We pick a random vector for each ingredient:"

# ╔═╡ d3c07d85-90e0-43ec-8468-b7db81629754
md"""
Here, we treat every ingredient as an independent entity. The dataset also contains information on the type of ingredient (fruit, vegetable, dairy...) and its flavour profile. If we feel like it, we could incorporate this information to make a better representation. We can also do some clever things to make ingredients that co-occur in recipes more similar. 

Given the HDVs of the ingredients, we can represent a recipe as the bundling of the composing ingredient representations. 
"""

# ╔═╡ f32fe25f-e45f-478a-a540-f406e37390da
md"""
Again, this is only the most basic thing we can do! We could also use binding to encode interactions between ingredients, which is, of course, hugely important to have a realistic cooking model.

Finally, we can bundle all 55,000 recipes in a single vector representing the distribution of recipes:
"""

# ╔═╡ 31d654a5-4258-46d9-a61b-1bbde14c21df
md"We can also do this for each of the origins separately. This collects information on ingredient use per region:"

# ╔═╡ 71462659-b977-4323-8ab0-00800ff662bc
md"Let us take a look at how the regions differ in recipe composition!"

# ╔═╡ b92c8919-0051-4be7-8a89-5ae5ccdd2b12
md"A simple PCA gives a graphical representation."

# ╔═╡ df50f82c-f2d1-4a64-854d-8702c313da38
md"""
What are the most essential ingredients for each region? We can match the embedding of each ingredient with the embedding of each region with the cosine similarity. Take a look at the top 10 ingredients per region below.
"""

# ╔═╡ f294ddf4-d6a3-47a0-b2b5-61592ebec2cc
md"We can create embeddings of new recipes and study them."

# ╔═╡ ae5e090e-9a81-4a3f-9c95-f2da12cf7562
md"Does this match the distribution of the true recipes?"

# ╔═╡ 64f6bc4d-73b2-4d5d-baf9-e41a867e44b4
md"As a reference, this is what we get if we match a random set of ingrdients to the master recipe embedding:"

# ╔═╡ b789899d-c95b-4a08-8051-25da7750e7d1
md"We can check which region matches best to our recipe."

# ╔═╡ 45cabcf5-a686-41e5-8b0d-cf55a62df956
md"And finally, we can look for the best ingredient that would complete our recipe. Just add ingredients, one at a time, and compare with the master recipes embedding."

# ╔═╡ 63d69906-35fc-4d19-8cce-d6b6e3774b8f
md"""
### Example 3: protein classification

As the last application, let us deal with some sequence data. To this end, we load 1000 protein sequences: half originating from humans, half from yeast. Can we build a simple classifier to discriminate between the species?
"""

# ╔═╡ 1e178b1c-9ae4-4dff-b140-c37a70d74fac
md"For our purposes, a protein just a sequence of amino acids. Ignoring any physicochemical similarity between them, we assign a random HDV to each HDV."

# ╔═╡ f5e89e20-ef6d-49e0-8fbd-62b7d828d9cf
md"""
We will consider each sequence as a bag of trimers to encode the sequences. We can use binding to create trimer-HDVs from the character vectors, but we must be cautious of retaining the order information since `bind(x, y, z)` is the same as `bind(y, x, z)`. Binding is commutative. We can encode the position information using shifting:

`bind(x, shift(y), shift(shift(z)))`

As there are only $20\times 20 \times 20=8000$ trimers, we can precompute them.
"""

# ╔═╡ 9e3279d9-5769-4b50-83d5-72c77ead7952
md"Then we look at all trimers in a sequence and bundle them."

# ╔═╡ ad4b7552-d7d2-4f7f-aea2-c39d6bcbf80f
asequence = "pelican"

# ╔═╡ b9a758b2-3522-4f5d-8403-b82ff321f9df
for i in 1:length(asequence)-2
	println(" "^(i-1) * asequence[i:i+2])
end 

# ╔═╡ d89e69ed-7c14-47d7-8aa8-dd07dfa7da18
md"We can take a look at the similarity between sequences and represent this in 2D using PCA."

# ╔═╡ 9008144b-a78c-426e-9cd2-366618cd8bd2
md"Let us build a simple classifier to predict the species. We take 80% of the data for training and 20% for testing. As the data is balanced, it is easy to compare. Let us explore three strategies to 'train' a classifier."

# ╔═╡ 0c5fce59-684c-4a3a-a1f9-795b989d5aa3
train = rand(n_seqs) .< 0.8;  # take 80% of the sequences for training

# ╔═╡ 9c0ab3a8-d5d8-4f50-8a16-1d6c9c8582d4
test = .! train;

# ╔═╡ b4394de6-cd0d-4a9b-8350-8433f3b43b4f
md"We can make prototypes for both species."

# ╔═╡ 06eeaea3-44ec-4cc5-8d8a-5f8db17cfefd
md"**Strategy 1**: is a new protein sequence closer to the human of yeast prototype?"

# ╔═╡ 97624ac0-0f33-4798-8f78-70a3ddbc6f0e
md"""
Not too bad for such a simple model. Most applications that use HDC for machine learning use some form of *retraining* they look at datapoints that were wrongly classified and use them to improve the prototypes. Again, here we won't bother. Instead, let us look at a *discriminative* approach.

**Strategy 2**: subtract the sequence embedding of yeast from those of humans to get a difference vector.
"""

# ╔═╡ 202f239b-ee55-4fbc-b36b-30e0a409ada9
md"""
We see that this is a slight improvement. Let us fit a slightly more clever model, using weights for the individual bits.

**Strategy 3**: fit a Naive Bayes model.

Given the label, we assume that all the HDV bits are independent. Because the information is distributed over the whole vector, it is not a strange assumption to make. For any given bit, we can estimate the probability of it being positive, given that it is either human or yeast. Computing the log ratio for every position

$$\log\left(\frac{\frac{p^+_{human}}{1-p^+_{human}}}{\frac{p^+_{yeast}}{1-p^+_{yeast}}}\right)\,,$$

Where $p^+_{human}$ is the probability that the bit is positive at a specific position in the human HDVsfor humans. Adding all these quantities for the different elements should give a pretty good global log-likelihood ratio.
"""

# ╔═╡ 959010fe-9ae1-4a47-8e62-1b867a8c3b9b
md"""
The Naive Bayes approach seems to work even better here!

We can look at which trimers agree best with our weight vector. This might be useful the discover which properties are important for making the prediction.
"""

# ╔═╡ 3969cdaa-0f7a-47f7-9061-b8bc35fc103b
md"Again, one can add many improvements to our sequence model."

# ╔═╡ f91210a1-fa3f-4894-ae9d-e15ec62439c6
md"""
## Comparision with other methods

### With deep learning

HDC seems to complement deep learning quite well. For 1D sequence-like data, HDC has been shown to outcompete convolutional neural networks. The latter still delivers superior performance for 2D image-like data at the expense of much more computation. I got intrigued by HDC by [a paper](https://arxiv.org/abs/2106.02894) that showed it works about as good as graph convolutional neural networks for predicting drug-like properties while taking mere minutes to train. HDC is an energy-efficient alternative to deep learning, with potential in fields such as [robotics](https://www.science.org/doi/10.1126/scirobotics.aaw6736).

HDC allows for more straightforward incorporation of prior knowledge and can handle specific problems more efficiently, such as the multi-instance setting. It is also reasonably data-efficient. I think it has tremendous potential for challenges such as causal inference.
"""

# ╔═╡ 6d82ba5f-40db-4df8-af41-0587cedacd74
md"""
### With kernel methods

HDC evokes strong parallels to kernel-based methods. Both work by representing the objects in some high-dimensional feature space. Kernel methods perform an *implicit* feature mapping to Hilbert space, accessed through dot-products in this space. One obtains powerful nonlinear models by replacing dot products with kernel evaluations in linear learning algorithms such as SVM or PCA.

HDC creates *explicit* and often discrete feature spaces. Rather than relying on linear algebra and linear learning methods, one typically uses nearest-neighbour-like algorithms, prototypes and the like. However, this seems to be more of a cultural choice, as the former can certainly be used in tandem with HDVs. The explicit feature space makes HDVs easier to manipulate than objects in Hilbert space. An added advantage (at least in my point of view) is that HDC is much easier to explain to the non-expert. HDC seems to shine for discrete problems and is slightly clumsy in modeling real-valued objects compared to kernel methods.
"""

# ╔═╡ d6ddd663-8069-4b1c-8af9-2699411343ea
md"""
### RandNLA

Hyperdimensional computing also evokes the ideas discussed in [randomized linear algebra](https://cacm.acm.org/magazines/2016/6/202647-randnla/fulltext) (RandNLA), which uses random sampling and projection to make certain linear algebra computations such as solving a system or computing the eigenvalues easier. RandNLA also distributes the information over all the matrix dimensions, improving their properties, such as the condition number. Matrices containing HDVs are likely to be well-conditioned, as all the vectors "look the same", having very similar norms etc.
"""

# ╔═╡ 5f584667-d3f0-4865-ac7c-23db4cdc7b71
md"""
## Further reading

> Kanerva, P. (2009) *Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors* Cognitive Computation

A great introduction to the idea and how to compute with HDVs. The author's [slide deck](https://web.stanford.edu/class/ee380/Abstracts/171025-slides.pdf) is very accessible.

> [Hyperdimensional computing and its role in AI](https://medium.com/dataseries/hyperdimensional-computing-and-its-role-in-ai-d6dc2828e6d6)

An outstanding blog post.

> Hassan et al. (2021) *Hyper-dimensional computing challenges and opportunities for AI applications* IEEE Access

Recent review and comparison with deep learning.

> Mitrokhin et al. *Learning sensorimotor control with neuromorphic sensors: Toward hyperdimensional active perception* Science Robotics

Case study of HDC in robotics. Though I am not really convinced of their scheme for encoding semantic meaning in the HDV, it explains the basic concepts of HDC quite well.

"""

# ╔═╡ 8111b866-9233-457d-ac6f-68b5dd766afa
md"""
## Appendix

This appendix contains some helper functions and code to generate the three examples.
"""

# ╔═╡ 9899dc83-4354-4217-9c0e-bbaa3db976b2
"""Get the top-10 elements from a list of (key, value), by value."""
top10(list) = sort!(list, rev=true, by=t->t[2])[1:10];

# ╔═╡ 7548130b-952b-41ab-8245-22da81a3c6be
md"""
### Generating the colour data
"""

# ╔═╡ fe25d401-c7f2-4657-bdad-ec393f0bcd5e
"""Randomly draw a color from the RGB-space"""
randcol() = RGB(rand(), rand(), rand());

# ╔═╡ b4655d6c-2169-4336-86c4-f97d55eca319
acolor = randcol()

# ╔═╡ a1861373-574f-452b-9a85-2772ddf0e585
acolor.r, acolor.g, acolor.b

# ╔═╡ c55f2c0c-f643-4ca8-8060-0a56f67cd1e8
colhdv = encode_col(acolor)

# ╔═╡ 5d3da98f-9f6c-4f8b-9229-7613cc460a2e
const ref_colors = [randcol() for i in 1:1000] .|> c -> (color=c, hdv=encode_col(c))

# ╔═╡ 425a47da-c33d-4f52-9f9e-e1429129fb77
decode_colors(v) = argmax(((c, cv),) -> cos(v, cv), ref_colors) |> first

# ╔═╡ 3a762ac8-4d6a-48a3-ac32-dd1f488aacfe
decode_colors(colhdv)  # we more or less recover the color!

# ╔═╡ 291b6f8c-038f-4a1c-a8d8-69d0a99a17de
decode_colors(hdv())  # decoding a random HDV

# ╔═╡ 75c22e04-c70a-43b0-a0f2-87cd6053adaa
begin
	# collect all colors
	reds = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("red", n)]
	blues = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("blue", n)]
	greens = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("green", n)]
	oranges = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("orange", n)]
	greys = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("grey", n)]
	yellows = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("yellow", n)]
	whites = [RGB((c./255)...) for (n, c) in Colors.color_names if occursin("white", n)]
end;

# ╔═╡ d1f8305e-fc31-4f16-904b-44e409c570c5
emojis_colors = Dict(:🚒 => reds, :💦 => blues, :🌱 => greens, :🌅 => oranges, :🐺 => greys, :🍌 => yellows, :🥚 => whites)

# ╔═╡ 93e2398c-f423-4e67-a8e1-87f0b304ff83
emojis = collect(keys(emojis_colors))

# ╔═╡ 4b63258f-c759-4f89-8949-272d7b596bc0
md"Encoding emojis is easy! There are only $(length(emojis)) of them. Let us give them all a unique HDV."

# ╔═╡ bf0286e1-ec61-4799-b068-8286916290d5
emojis

# ╔═╡ d5f14066-1773-4dd7-93fd-d8d70bba6c04
emojis_hdvs = Dict(s => hdv() for s in emojis)

# ╔═╡ 42a4f962-a3f4-464c-9af7-abe1168738cb
encode_emoji_col_pair((s, c)) = bind(emojis_hdvs[s], encode_col(c))

# ╔═╡ e9135797-b217-4833-8d69-b06615ebb10c
encode_shape_col_pair2((s, colors)) = bind(emojis_hdvs[s], bundle(encode_col.(colors)...))

# ╔═╡ 33187ee1-26ca-445a-8dd4-e6c752e4ddcd
toy_data1 = [rand(emojis) |> l->(l, rand(emojis_colors[l])) for i in 1:100]

# ╔═╡ 845d7044-e9ae-4e83-bc39-969cd49405e4
toy_data1

# ╔═╡ eb2e1852-79f9-45fe-a6f9-818e56aa0f5f
col_emoji_hdvs = encode_emoji_col_pair.(toy_data1)

# ╔═╡ 3d50923f-e5c0-4a8c-809c-4856068756b9
toy_data_emb = bundle(col_emoji_hdvs...)

# ╔═╡ c6a50c5f-3702-4c64-b47b-925189ebcfbe
bind(toy_data_emb, emojis_hdvs[:🚒])  # embedding of color of the truck

# ╔═╡ c021ab9a-a097-4994-917b-74c767189c2d
decode_colors(bind(toy_data_emb, emojis_hdvs[:🚒]))

# ╔═╡ 7bf217f7-639b-46c7-8565-4cdbf82bba26
decode_colors(bind(toy_data_emb, emojis_hdvs[:💦]))

# ╔═╡ 4c557c67-aa46-416e-a31a-c791dc954a9d
decode_colors(bind(toy_data_emb, emojis_hdvs[:🌱]))

# ╔═╡ ac580e2a-c2bf-4243-8502-2263a4e8d1f4
decode_colors(bind(toy_data_emb, emojis_hdvs[:🍌]))

# ╔═╡ 30c0e722-9457-4609-943f-c7b837b45151
toy_data2 = [rand(emojis) |> l->(l, shuffle!([rand(emojis_colors[l]), randcol(), randcol()])) for i in 1:500]

# ╔═╡ 4c417b6d-97ad-4176-bb7f-4ccb14bdc6cf
toy_data2

# ╔═╡ eb1e08c1-72e6-46dd-936e-0f7b09dbaa29
toy_data_emb2 = bundle(encode_shape_col_pair2.(toy_data2)...)

# ╔═╡ 57593d22-5be6-484a-93b9-323ae0d982cc
decode_colors(bind(toy_data_emb2, emojis_hdvs[:🚒]))

# ╔═╡ 39ce810c-bcc8-4ed8-b349-eef5c11b1b1a
decode_colors(bind(toy_data_emb2, emojis_hdvs[:💦]))

# ╔═╡ 860deb07-0ddd-4b04-96c8-e67bd3f5e5b6
decode_colors(bind(toy_data_emb2, emojis_hdvs[:🌱]))

# ╔═╡ ecbcf732-9459-425f-aa51-be9d5a32667a
decode_colors(bind(toy_data_emb2, emojis_hdvs[:🍌]))

# ╔═╡ c1878a27-fb4e-4315-a922-1e3f45844b32
md"""
### Loading the recipes data
"""

# ╔═╡ f642e17c-ef79-4fd0-86df-5155bb44284c
recipes_data = CSV.read("data/Recipes_with_origin.csv", DataFrame)[:,2:end];

# ╔═╡ 4284e904-5541-4568-bc9f-0547f65741e5
recipes_data

# ╔═╡ d8405623-6eef-45d4-857d-5bf9793957e5
begin
	ingredients = names(recipes_data)[1:end-11]
	n_recipes = size(recipes_data, 1)

	ingr2ind = Dict(ingr=>i for (i,ingr) in enumerate(ingredients))

	origins = recipes_data[:,end-10:end]
	regions = names(origins)
	
	recipes = sparse(Matrix(recipes_data[:,1:end-11]))
end;

# ╔═╡ d597efc2-ce62-45bc-b0e5-9818730566e2
ingredients

# ╔═╡ 026a628a-f9fd-44ac-a2cb-814c2271a8c4
Xingr = hdv(length(ingredients))

# ╔═╡ 29b61795-178e-4258-b735-79b29abb8d72
Xrec = recipes * Xingr .|> sign

# ╔═╡ 1b94e0dc-3dc7-43a0-bb5c-58bcf53ccc00
recipes_hdv = sum(Xrec, dims=1)[:] .|> sign

# ╔═╡ 371a69e4-7e1c-42c4-b604-eef6702c699b
Xorigin = Matrix(origins)' * Xrec .|> sign

# ╔═╡ 7fc3ebac-0e69-4f8b-b140-daa726a67f60
C_or = Xorigin * Xorigin';

# ╔═╡ e45bb41b-aa53-402d-99ac-332dd5be4287
DataFrame([regions C_or], ["origin", regions...])

# ╔═╡ 2bf5b6f7-4d2b-4610-aace-d8b0da82e670
let
	# simple PCA
	C̃ = C_or .-  mean(C_or, dims=1) .- mean(C_or, dims=2) .+ mean(C_or)
	Λ, V = eigen(C̃)	
	
	p = scatter(V[:,end], V[:,end-1], label="", color="black")

	annotate!(p, V[:,end], V[:,end-1], regions)
	p
end

# ╔═╡ 081e33de-c308-416c-b081-1a564f52089f
DataFrame(
	Dict(r => [(ingr, cos(Xingr[i,:], Xorigin[j,:])) for (i, ingr) in enumerate(ingredients)] |> top10
			for (j, r) in enumerate(regions)))

# ╔═╡ f233f712-832d-474d-8994-bdd0c478cab1
embed_ingr(ingr) = Xingr[ingr2ind[ingr], :]

# ╔═╡ 8cde2df6-8259-4884-b0fd-aac1178077a5
encode_recipe(recipe) = bundle(embed_ingr.(recipe)...)

# ╔═╡ 9ae32d1b-1e06-4ddf-9db2-f9fbb2cbbbbc
my_recipe_hdv = encode_recipe(["wine", "butter", "lemon peel", "chicken", "black pepper", "cheese"])

# ╔═╡ 4ae2c7ce-602b-4fce-9d39-5d4adfd1a584
cos(my_recipe_hdv, recipes_hdv)

# ╔═╡ 8317b9af-dcb3-45bf-8419-77d6d422a4eb
rand_ingr = rand(ingredients, 5)

# ╔═╡ b46651f5-2e8d-4d3c-bb84-8a85f1d788f0
fake_rec_hdv = encode_recipe(rand_ingr)

# ╔═╡ 63da4716-384a-470f-9758-cd3c46229813
cos(fake_rec_hdv, recipes_hdv)  #  lower!

# ╔═╡ e87efdc5-c0f2-40c8-b5d0-476888b91ef6
zip(regions, Xorigin * my_recipe_hdv) |> collect |> top10

# ╔═╡ cadfce71-ec41-440d-ac06-ce2e4fa59818
[(ingr, cos(recipes_hdv,  Xingr[i,:] .* my_recipe_hdv)) for (i, ingr) in enumerate(ingredients)] |> top10

# ╔═╡ cfb7d15f-7731-4b7e-8868-e6d428af0251
md"""
### Loading the protein data
"""

# ╔═╡ 4bd3b204-c86f-4db1-818b-7cc3b239ac1c
amino_acids = mapreduce(unique, union, sequences) |> sort!

# ╔═╡ 3382d97d-7df0-4fd3-9243-104a61db8335
amino_acids

# ╔═╡ 167861f2-0ad7-444a-89d7-762e34f5fe00
const aa_hdvs = Dict(aa=>hdv() for aa in amino_acids)

# ╔═╡ 7b030f8e-9a10-438a-b702-9b29ee0250c3
const trimer_hdvs = Dict(aa1 * aa2 * aa3 => 
					bind(aa_hdvs[aa1], shift(aa_hdvs[aa2]), shift(aa_hdvs[aa3], 2))
			for aa1 in amino_acids for aa2 in amino_acids for aa3 in amino_acids)

# ╔═╡ 1fc29d2b-4694-4794-8191-3d4dfbbfcbf7
function embed_sequences(sequences)
	# preallocate an empty matrix
	hdvs = zeros(Int, length(sequences), N)
	for (i, seq) in enumerate(sequences)
		v = @view hdvs[i,:]  # ref to hdv i
		for pos in 1:length(seq)-2
			trimer = seq[pos:pos+2]
			v .+= trimer_hdvs[trimer]
		end
		v .= sign.(v)
	end
	return hdvs
end

# ╔═╡ 40605a9d-5fcd-45e4-b953-72b618fc239b
Xseq = embed_sequences(sequences)

# ╔═╡ c7a54d09-d61f-4e6b-bf53-4732d3b58e09
let
	Cseq = Xseq * Xseq'
	# simple PCA
	C̃ = Cseq .-  mean(Cseq, dims=1) .- mean(Cseq, dims=2) .+ mean(Cseq)
	Λ, V = eigen(C̃)	
	
	p = scatter(V[humane,end], V[humane,end-1], label="human", color="blue", alpha=0.7)
	scatter!(V[.!humane,end], V[.!humane,end-1], label="yeast", color="orange", alpha=0.7)
	p
end

# ╔═╡ 43586fa7-2e82-4eac-a330-fa9acefb18dc
human_hdv = bundle(Xseq[train.&humane,:])

# ╔═╡ 7e4bf5ae-05cd-4e93-9b6f-daa0b71d89d7
yeast_hdv = bundle(Xseq[train.&.!humane,:])

# ╔═╡ 204bb552-9cb3-447c-820d-c0030db016c9
cos(human_hdv, yeast_hdv)  # prototypes are related, as would be expected

# ╔═╡ 0833f9bd-0d3f-4379-9a84-8c8895b5f86c
predict_sp(x) = cos(human_hdv, x) > cos(yeast_hdv, x) ? "HUMAN" : "YEAST"

# ╔═╡ 540ff878-8b30-4f39-b8c6-8de47938c082
predictions = predict_sp.(eachrow(Xseq[test,:]))

# ╔═╡ 83542647-f2fb-4cd8-b7ac-fcbd18e99629
mean(species[test] .== predictions)

# ╔═╡ 563f0d4e-176b-4e0f-a3a5-f3e2b30ccd4f
hdv_diff = sum(Xseq[train.&humane,:], dims=1)[:] -  
			sum(Xseq[train.&.!humane,:], dims=1)[:] #.|> sign

# ╔═╡ b745a8c1-1b40-4e89-a2f2-6023c1224b4f
predict_sp2(x) = cos(hdv_diff, x) > 0 ? "HUMAN" : "YEAST"

# ╔═╡ ee157137-78c1-4cfe-a5b1-39408d05b900
predictions2 = predict_sp2.(eachrow(Xseq[test,:]))

# ╔═╡ 43218d25-082e-44e4-bd06-d1eaf7ae5dc4
mean(species[test] .== predictions2)

# ╔═╡ 312c04ef-9a88-4f2e-ac9d-0367e74c1b2c
P₊human = mean(>(0), Xseq[train.&humane,:], dims=1)[:]

# ╔═╡ b81e0880-90f1-4494-a537-a8a521937064
P₊yeast = mean(>(0), Xseq[train.&.!humane,:], dims=1)[:]

# ╔═╡ 271c915d-050c-410a-9a87-52762e62f9c9
θ = @. (log(P₊human) - log(1-P₊human)) - (log(P₊yeast) - log(1-P₊yeast))

# ╔═╡ 51f12893-09e7-4483-b558-b5d6450720cd
predict_sp3(x) = dot(θ, x) > 0 ? "HUMAN" : "YEAST"

# ╔═╡ 7fd5b9bd-3e7e-41b0-9a7d-a6bb6cc4220d
predictions3 = predict_sp3.(eachrow(Xseq[test,:]))

# ╔═╡ 2c80eacd-2265-40e3-a2a6-80977e15af66
mean(species[test] .== predictions3)

# ╔═╡ fc14fadf-8bea-4a33-9faf-7149edba5db9
[(trimer, cos(v, θ)) for (trimer, v) in trimer_hdvs] |> top10

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
FASTX = "c2308a5c-f048-11e8-3e8a-31650f418d12"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.4"
Colors = "~0.12.8"
DataFrames = "~1.3.4"
FASTX = "~2.0.0"
Plots = "~1.32.0"
PlutoUI = "~0.7.40"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BioGenerics]]
deps = ["TranscodingStreams"]
git-tree-sha1 = "0b581906418b93231d391b5dd78831fdc2da0c82"
uuid = "47718e42-2ac5-11e9-14af-e5595289c2ea"
version = "0.1.2"

[[BioSequences]]
deps = ["BioSymbols", "Random", "Twiddle"]
git-tree-sha1 = "523d40090604deae32078e3bf3d8570ab1cb585b"
uuid = "7e6ae17a-c86d-528c-b3b9-7f778a29fe59"
version = "3.1.0"

[[BioSymbols]]
git-tree-sha1 = "6f59deb6e86841a75188721c567fad81fbc305f1"
uuid = "3c28c6f8-a34d-59c4-9654-267d177fcfa9"
version = "5.1.1"

[[BitFlags]]
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dc4405cee4b2fe9e1108caec2d760b7ea758eca2"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.5"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[FASTX]]
deps = ["Automa", "BioGenerics", "BioSequences", "ScanByte", "StringViews", "TranscodingStreams"]
git-tree-sha1 = "9c72011edd523a83bf00276c4697a3019cea2257"
uuid = "c2308a5c-f048-11e8-3e8a-31650f418d12"
version = "2.0.0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0eb5ef6f270fb70c2d83ee3593f56d02ed6fc7ff"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.68.0+0"

[[GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "12a584db96f1d460421d5fb8860822971cdb8455"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.4"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "4abede886fcba15cd5fd041fef776b230d004cee"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.4.0"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "fa44e6aa7dfb963746999ca8129c1ef2cf1c816b"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.1.1"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "e9cab2c5e3b7be152ad6241d2011718838a99a27"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.32.1"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "e71ccdc4a444d50b2cabd807ad77693bd423b14c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.41"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "130c68b3497094753bacf084ae59c9eeaefa2ee7"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.14"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "efa8acd030667776248eabb054b1836ac81d92f0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.7"

[[StaticArraysCore]]
git-tree-sha1 = "ec2bd695e905a3c755b33026954b119ea17f2d22"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.3.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[StringViews]]
git-tree-sha1 = "ac5ab9fa38d95857460e2f988d1a5c0de1ff34e3"
uuid = "354b36f9-a18e-4713-926e-db85100087ba"
version = "1.0.2"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "7149a60b01bf58787a1b83dad93f90d4b9afbe5d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.8.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[Twiddle]]
git-tree-sha1 = "29509c4862bfb5da9e76eb6937125ab93986270a"
uuid = "7200193e-83a8-5a55-b20d-5d36d44a0795"
version = "1.1.2"

[[URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─f6b4fc92-51fa-42ba-8f3e-6f81bc6d8624
# ╟─a3820f10-9930-4dba-85aa-80a09744578d
# ╠═366fc4c8-2d15-11ed-18cd-c14f15c96f6e
# ╠═d3f992f2-bd1e-4544-809f-b9fab1c11005
# ╟─59f5ad60-4554-462e-9a7b-1b7e465b720b
# ╟─0424b37b-64f6-48c6-8e61-85b6e62fc93c
# ╠═09efbacf-8ee9-46c1-a4a7-11921fbc783b
# ╟─b5358dae-d136-4343-a3d9-d58ab5f1a79c
# ╠═0418d5fd-df04-4d56-ba7c-d45d18aa161d
# ╠═67dc0274-ae49-4262-ab9d-a7716ab7e851
# ╠═8123aaec-aa78-495e-83a1-3a4f4b244000
# ╟─49018537-b529-4d07-b1cb-d34fe58294f0
# ╠═ea146895-6d82-4dcb-a24d-0dae70ae39b9
# ╟─0c72733c-897b-4d9b-84f7-98efe2f197de
# ╟─f34d034a-942b-41a0-92ec-23608a43f778
# ╠═4c647ad6-3e70-48f6-b778-8f9a34d58a6f
# ╠═7ea554a1-5efb-461f-9654-2abff0dc80b5
# ╟─1a7bbd06-b6d8-4e0a-a6d7-cb082b9bb69d
# ╠═a40b8155-34f6-47cc-aaa0-82069ca2dfb5
# ╠═eebcab6e-a7e9-4cfd-9e6e-3e39c5d4b8f6
# ╠═119cd151-26b8-4d4e-a919-88c54cac8616
# ╠═adf79e3a-26f3-4afe-bd14-fc48880a29ec
# ╟─3578bb7d-353d-4f48-9f50-50b4aaaad19d
# ╠═f8170a5b-4c19-4c64-ad3a-ab82a8b448c6
# ╠═591e5938-3bba-4b72-8d20-0848884e732b
# ╟─3b543c80-8f36-49e9-a2fc-a80b6677ad1e
# ╠═9d2ceb8b-61ea-49c5-b3a4-08b061423e7c
# ╟─66440162-3b60-4078-bdcf-3426403fcc2f
# ╠═4e956ea8-fd2d-4fdb-be33-f8987b2bd1e5
# ╠═b2dc53a5-0bcd-4551-bf10-9ce405e8b651
# ╠═abab4185-4b15-4457-aa63-3c825477ae48
# ╟─a1eedfc9-4ee4-4f0c-8f89-1fe70b71a686
# ╠═8b153389-1da3-47b4-b747-ec252e08d90d
# ╠═59425927-abaa-45a9-a5e8-1619ae806840
# ╟─c5a8dd2f-b134-4b7b-96f3-3ad3611d92a4
# ╠═3bd2b8c8-7a57-4080-8861-f231dfa4471e
# ╠═3ce916d7-935d-43ee-b5dc-10651f281985
# ╟─efab3c5b-bb21-486e-8474-6b8440c9e42b
# ╠═a7d1326e-250c-4fd7-af2b-2f5e0504f03e
# ╟─8fc6f3a0-e05d-44cc-b3e3-bf35c1d02918
# ╠═432a9fae-1c68-4312-8837-9fe489e26791
# ╟─b5e16410-82eb-4851-a20d-92106c2654e9
# ╟─001f44f6-7e3f-4312-8387-9587f7811fdb
# ╟─84a318f7-3d7c-4809-91ce-d3388f915b3f
# ╟─27e6169f-1bfa-4e59-bf0f-872a432983d5
# ╟─c6718aac-f7e9-47b7-9698-d52c31208db9
# ╠═845d7044-e9ae-4e83-bc39-969cd49405e4
# ╟─4b63258f-c759-4f89-8949-272d7b596bc0
# ╠═bf0286e1-ec61-4799-b068-8286916290d5
# ╠═d5f14066-1773-4dd7-93fd-d8d70bba6c04
# ╟─853a5ec0-4a9a-4430-b0f2-b3e7994c25ba
# ╠═b4655d6c-2169-4336-86c4-f97d55eca319
# ╠═a1861373-574f-452b-9a85-2772ddf0e585
# ╟─a193a721-3caa-4069-b53f-3cdb58cb075e
# ╠═ec7cfa4e-d394-4b2a-9a4a-160c080fa341
# ╟─16e9a27f-0275-48b3-bbc3-9394725ec595
# ╠═c6ecd79c-0285-47a2-9c33-af440f5d3325
# ╠═0a1227c0-6557-46e7-8dcb-e7867b1aac94
# ╟─0f174703-27df-4c07-a03d-95cf482a1c1d
# ╟─7976d118-a1e8-43d8-862b-a21a05a7306e
# ╟─6aa8efb1-8705-474d-bd79-a75d78643f6c
# ╠═9c2b8b28-751d-4def-ae1c-9b1bf545f23f
# ╠═d751b589-8c7d-48d5-9548-3fe105b969e8
# ╟─909a155e-5cd0-4360-ac9c-4761521293db
# ╠═51e5be0b-4dfa-40c3-bbda-bd3b44595c86
# ╠═5eb33b1a-f91e-481c-893b-7e179a8136e5
# ╟─3de9f13e-4a6f-4ba0-b2d4-86fa617c7317
# ╠═c55f2c0c-f643-4ca8-8060-0a56f67cd1e8
# ╟─aab48f4b-10da-4872-aee2-bec419f867d2
# ╠═5d3da98f-9f6c-4f8b-9229-7613cc460a2e
# ╠═425a47da-c33d-4f52-9f9e-e1429129fb77
# ╠═3a762ac8-4d6a-48a3-ac32-dd1f488aacfe
# ╠═291b6f8c-038f-4a1c-a8d8-69d0a99a17de
# ╟─3f51851d-bfbd-4301-880f-923851305bf5
# ╠═42a4f962-a3f4-464c-9af7-abe1168738cb
# ╠═eb2e1852-79f9-45fe-a6f9-818e56aa0f5f
# ╠═3d50923f-e5c0-4a8c-809c-4856068756b9
# ╟─93c47ae3-1fcf-470f-90c6-28dad06631ad
# ╠═c6a50c5f-3702-4c64-b47b-925189ebcfbe
# ╠═c021ab9a-a097-4994-917b-74c767189c2d
# ╠═7bf217f7-639b-46c7-8565-4cdbf82bba26
# ╠═4c557c67-aa46-416e-a31a-c791dc954a9d
# ╠═ac580e2a-c2bf-4243-8502-2263a4e8d1f4
# ╟─8e334f45-08d0-455b-9a99-a1978cb9ae61
# ╠═4c417b6d-97ad-4176-bb7f-4ccb14bdc6cf
# ╟─4ede9c89-a64a-40f8-898a-22b7edb7d616
# ╠═e9135797-b217-4833-8d69-b06615ebb10c
# ╠═eb1e08c1-72e6-46dd-936e-0f7b09dbaa29
# ╠═57593d22-5be6-484a-93b9-323ae0d982cc
# ╠═39ce810c-bcc8-4ed8-b349-eef5c11b1b1a
# ╠═860deb07-0ddd-4b04-96c8-e67bd3f5e5b6
# ╠═ecbcf732-9459-425f-aa51-be9d5a32667a
# ╟─02f703ff-f44f-445d-bfb6-9aa3273c0393
# ╟─00b949c3-5986-4f72-a1db-45f5fa320d91
# ╠═4284e904-5541-4568-bc9f-0547f65741e5
# ╟─2df88a19-6965-4e48-92f9-3725ff86f7ad
# ╠═d597efc2-ce62-45bc-b0e5-9818730566e2
# ╟─6f8d9c88-1686-4a54-9cf6-ce53c31629a8
# ╠═026a628a-f9fd-44ac-a2cb-814c2271a8c4
# ╟─d3c07d85-90e0-43ec-8468-b7db81629754
# ╠═29b61795-178e-4258-b735-79b29abb8d72
# ╟─f32fe25f-e45f-478a-a540-f406e37390da
# ╠═1b94e0dc-3dc7-43a0-bb5c-58bcf53ccc00
# ╟─31d654a5-4258-46d9-a61b-1bbde14c21df
# ╠═371a69e4-7e1c-42c4-b604-eef6702c699b
# ╟─71462659-b977-4323-8ab0-00800ff662bc
# ╠═7fc3ebac-0e69-4f8b-b140-daa726a67f60
# ╠═e45bb41b-aa53-402d-99ac-332dd5be4287
# ╟─b92c8919-0051-4be7-8a89-5ae5ccdd2b12
# ╟─2bf5b6f7-4d2b-4610-aace-d8b0da82e670
# ╟─df50f82c-f2d1-4a64-854d-8702c313da38
# ╠═081e33de-c308-416c-b081-1a564f52089f
# ╟─f294ddf4-d6a3-47a0-b2b5-61592ebec2cc
# ╠═f233f712-832d-474d-8994-bdd0c478cab1
# ╠═8cde2df6-8259-4884-b0fd-aac1178077a5
# ╠═9ae32d1b-1e06-4ddf-9db2-f9fbb2cbbbbc
# ╟─ae5e090e-9a81-4a3f-9c95-f2da12cf7562
# ╠═4ae2c7ce-602b-4fce-9d39-5d4adfd1a584
# ╟─64f6bc4d-73b2-4d5d-baf9-e41a867e44b4
# ╠═8317b9af-dcb3-45bf-8419-77d6d422a4eb
# ╠═b46651f5-2e8d-4d3c-bb84-8a85f1d788f0
# ╠═63da4716-384a-470f-9758-cd3c46229813
# ╟─b789899d-c95b-4a08-8051-25da7750e7d1
# ╠═e87efdc5-c0f2-40c8-b5d0-476888b91ef6
# ╟─45cabcf5-a686-41e5-8b0d-cf55a62df956
# ╠═cadfce71-ec41-440d-ac06-ce2e4fa59818
# ╟─63d69906-35fc-4d19-8cce-d6b6e3774b8f
# ╟─1e178b1c-9ae4-4dff-b140-c37a70d74fac
# ╠═3382d97d-7df0-4fd3-9243-104a61db8335
# ╠═167861f2-0ad7-444a-89d7-762e34f5fe00
# ╟─f5e89e20-ef6d-49e0-8fbd-62b7d828d9cf
# ╠═7b030f8e-9a10-438a-b702-9b29ee0250c3
# ╟─9e3279d9-5769-4b50-83d5-72c77ead7952
# ╠═ad4b7552-d7d2-4f7f-aea2-c39d6bcbf80f
# ╟─b9a758b2-3522-4f5d-8403-b82ff321f9df
# ╠═1fc29d2b-4694-4794-8191-3d4dfbbfcbf7
# ╠═40605a9d-5fcd-45e4-b953-72b618fc239b
# ╟─d89e69ed-7c14-47d7-8aa8-dd07dfa7da18
# ╟─c7a54d09-d61f-4e6b-bf53-4732d3b58e09
# ╟─9008144b-a78c-426e-9cd2-366618cd8bd2
# ╠═0c5fce59-684c-4a3a-a1f9-795b989d5aa3
# ╠═9c0ab3a8-d5d8-4f50-8a16-1d6c9c8582d4
# ╟─b4394de6-cd0d-4a9b-8350-8433f3b43b4f
# ╠═43586fa7-2e82-4eac-a330-fa9acefb18dc
# ╠═7e4bf5ae-05cd-4e93-9b6f-daa0b71d89d7
# ╠═204bb552-9cb3-447c-820d-c0030db016c9
# ╟─06eeaea3-44ec-4cc5-8d8a-5f8db17cfefd
# ╠═0833f9bd-0d3f-4379-9a84-8c8895b5f86c
# ╠═540ff878-8b30-4f39-b8c6-8de47938c082
# ╠═83542647-f2fb-4cd8-b7ac-fcbd18e99629
# ╟─97624ac0-0f33-4798-8f78-70a3ddbc6f0e
# ╠═563f0d4e-176b-4e0f-a3a5-f3e2b30ccd4f
# ╠═b745a8c1-1b40-4e89-a2f2-6023c1224b4f
# ╠═ee157137-78c1-4cfe-a5b1-39408d05b900
# ╠═43218d25-082e-44e4-bd06-d1eaf7ae5dc4
# ╟─202f239b-ee55-4fbc-b36b-30e0a409ada9
# ╠═312c04ef-9a88-4f2e-ac9d-0367e74c1b2c
# ╠═b81e0880-90f1-4494-a537-a8a521937064
# ╠═271c915d-050c-410a-9a87-52762e62f9c9
# ╠═51f12893-09e7-4483-b558-b5d6450720cd
# ╠═7fd5b9bd-3e7e-41b0-9a7d-a6bb6cc4220d
# ╠═2c80eacd-2265-40e3-a2a6-80977e15af66
# ╟─959010fe-9ae1-4a47-8e62-1b867a8c3b9b
# ╠═fc14fadf-8bea-4a33-9faf-7149edba5db9
# ╟─3969cdaa-0f7a-47f7-9061-b8bc35fc103b
# ╟─f91210a1-fa3f-4894-ae9d-e15ec62439c6
# ╟─6d82ba5f-40db-4df8-af41-0587cedacd74
# ╟─d6ddd663-8069-4b1c-8af9-2699411343ea
# ╟─5f584667-d3f0-4865-ac7c-23db4cdc7b71
# ╟─8111b866-9233-457d-ac6f-68b5dd766afa
# ╠═9899dc83-4354-4217-9c0e-bbaa3db976b2
# ╟─7548130b-952b-41ab-8245-22da81a3c6be
# ╠═fe25d401-c7f2-4657-bdad-ec393f0bcd5e
# ╟─75c22e04-c70a-43b0-a0f2-87cd6053adaa
# ╠═d1f8305e-fc31-4f16-904b-44e409c570c5
# ╠═93e2398c-f423-4e67-a8e1-87f0b304ff83
# ╠═33187ee1-26ca-445a-8dd4-e6c752e4ddcd
# ╠═30c0e722-9457-4609-943f-c7b837b45151
# ╟─c1878a27-fb4e-4315-a922-1e3f45844b32
# ╠═3cdc0adf-2ba6-4213-b644-907533de412f
# ╠═f642e17c-ef79-4fd0-86df-5155bb44284c
# ╠═d8405623-6eef-45d4-857d-5bf9793957e5
# ╟─cfb7d15f-7731-4b7e-8868-e6d428af0251
# ╠═7f6763c0-c440-4e9d-a514-98ec4273bd57
# ╠═4bd3b204-c86f-4db1-818b-7cc3b239ac1c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
