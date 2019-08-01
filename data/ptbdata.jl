using Printf: @printf
using Random: shuffle!
using Flux
using Flux: batchseq, OneHotVector, OneHotMatrix

# TODO(ying) Is it necessary to control the use of global varaiables?
const traindata = "ptb.train.txt"
const validdata = "ptb.valid.txt"
const testdata = "ptb.test.txt"
const vocabfile = "ptb.vocab.txt"

const UNK = "<unk>"
const PAD = "</p>"
const BOS = "<s>"
const EOS = "<e>"

"""
    ptb_download()

TODO(ying): Add docstring.
"""
function ptb_download()
  filename = "simple-examples"
  url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/$filename.tgz"
  savepath = joinpath(@__DIR__, "$filename.tgz")

  isfile(savepath) || download(url, savepath)
  cd(@__DIR__) do
    for file in [traindata, testdata, validdata]
      run(`tar --strip-components 3 -xzf $savepath $filename/data/$file`)
    end
  end
end

"""
    buildvocab(inputfile, savepath)

Read words from `inputfile` and count their frequencies, build and then save the
word vocabulary which contains all the words appear in `inputfile`.

The word dictionary is saved to `savepath`. Each line in `savepath` is a
word and its frequency in training corpus seperated by "\t". Three special
token <s> <e> and </p> is added to the begining of the dictionary.

NOTE: this function always builds a new word dictionary and save it to
`savepath`, regardless of whether `savepath` file exits.
"""
function buildvocab(inputfile, savefile)
  worddict = Dict{String, Int}()
  open(inputfile, "r") do fin
    for line in eachline(fin)
      for word in split(rstrip(line))
          worddict[word] = get!(worddict, word, 0) + 1
      end
    end
  end

  open(joinpath(@__DIR__, savefile), "w") do fdict
    # Add three special marks: <s> for start mark,
    # <e> for end mark and <p> for padding mark.
    foreach(x -> @printf(fdict, "%s\t-1\n", x), [BOS, EOS, PAD])
    foreach(x -> @printf(fdict, "%s\t%d\n", x[1], x[2]),
            sort(collect(worddict), by=x->x[2], rev=true))
  end
end

"""
    loadvocab(vocabfile)

TODO(ying): Add docstring.
"""
function loadvocab(vocabfile)
    worddict = Dict{String, Int}()
    open(vocabfile, "r") do fdict
        for (index, line) in enumerate(eachline(fdict))
            word, frequency = split(line, "\t")
            get!(worddict, word, index)
        end
    end
    return worddict
end

"""
    getvocab()

Return the word vocabulary for PTB dataset.
Build the word vocabulary if it does not exist.
"""
function getvocab()
  vocabpath = joinpath(@__DIR__, vocabfile)
  !isfile(vocabpath) && buildvocab(joinpath(@__DIR__, traindata), vocabpath)
  worddict = loadvocab(vocabpath)
end

# TODO(Ying): optimize the printing format making it more readable.
Base.show(io::IO, c::OneHotVector) = print(io, c.ix)
Base.show(io::IO, c::OneHotMatrix) = print(io, c.data)

onehotmatrix(batch, length) =
        OneHotMatrix(length, [OneHotVector(w, length) for w in batch])

"""
    getdatabatch(datafile; enable_shuffle=false)

TODO(ying): Add docstring.
"""
function getdatabatch(batchsize, datafile; enable_shuffle=false)
  datapath = joinpath(@__DIR__, datafile)
  isfile(datapath) || ptb_download()

  worddict = getvocab()
  @assert(haskey(worddict, UNK),
          "Word vocabulary should contain a <unk> token.")
  unkid = worddict[UNK]
  padid = worddict[PAD]

  dataset = Vector{Int}[]
  dictlength = length(worddict)
  open(datapath, "r") do fdata
    for (idx, line) in enumerate(eachline(fdata))
      push!(dataset, Int[])

      # Add a start mark at the begining of the sentence,
      # and add an ending mark at the end of the sentence.
      foreach(word -> push!(dataset[end], get!(worddict, word, unkid)),
              split("$BOS $line $EOS"))
    end
  end
  enable_shuffle && shuffle!(dataset)

  xs = Vector{<:OneHotMatrix}[]
  ys = Vector{<:OneHotMatrix}[]
  for batch in Iterators.partition(dataset, batchsize)
    maxlength = (length.(batch) |> maximum) - 1
    xs_ = batchseq([sample[1 : end - 1] for sample in batch], padid, maxlength)
    ys_ = batchseq([sample[2 : end] for sample in batch], padid, maxlength)

    push!(xs, [onehotmatrix(batch, dictlength) for batch in xs_])
    push!(ys, [onehotmatrix(batch, dictlength) for batch in ys_])
  end
  return zip(xs, ys)
end

function getvocabdim()
  vocabpath = joinpath(@__DIR__, vocabfile)
  !isfile(vocabpath) && buildvocab(joinpath(@__DIR__, traindata), vocabpath)
  return length(readlines(open(vocabpath)))
end

gettrainbatch(batchsize) = getdatabatch(batchsize,
        joinpath(@__DIR__, traindata); enable_shuffle=true)
getdevbatch(batchsize) = getdatabatch(batchsize, joinpath(@__DIR__, validdata);
        enable_shuffle=false)
gettestbatch(batchsize) = getdatabatch(batchsize, joinpath(@__DIR__, testdata);
        enable_shuffle=false)
