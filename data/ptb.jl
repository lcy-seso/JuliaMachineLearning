module Ptb

using Random: shuffle!
using Printf: @printf
using Flux: batchseq

using ..Data:download_and_verify

const URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
const SHA256 = "7d4fe4bd080182dd8423e32557cdbc3e5a967f7eb4a2a598be8ada7b66bd1de7"

const traindata = "ptb.train.txt"
const validdata = "ptb.valid.txt"
const testdata = "ptb.test.txt"
const vocabfile = "ptb.vocab.txt"

const UNK = "<unk>"
const PAD = "</p>"
const BOS = "<s>"
const EOS = "<e>"

"""
    ptb_download(savedir)

Download PTB dataset if it does not exist.
"""
function ptb_download(savedir)
    isdir(savedir) || mkdir(savedir)

    filename = "simple-examples"
    savepath = joinpath(savedir, "$filename.tgz")
    isfile(savepath) || download_and_verify(URL, savepath, SHA256)
    cd(savedir) do
        for file in [traindata, testdata, validdata]
            run(`tar --strip-components 3 -xzf $savepath ./$filename/data/$file`)
        end
    end
end

"""
    buildvocab(inputfile, savefile, savedir)

- inputfile: input data to build the word vocabulary.
- savefile: file name to save the built vocabulary.
- savedir: directory to save the built vocabulary.

Read words from `inputfile` and count their frequencies, build and then save the
word vocabulary which contains all the words appear in `inputfile`.

The word dictionary is saved to `savepath`. Each line in `savepath` is a
word and its frequency in training corpus seperated by "\t". Three special
token <s> <e> and </p> is added to the begining of the dictionary.

NOTE: this function always builds a new word dictionary and save it to
`savepath`, regardless of whether `savepath` file exits.
"""
function buildvocab(inputfile, savefile, savedir)
    datapath = joinpath(savedir, inputfile)
    isfile(datapath) || ptb_download(savedir)

    worddict = Dict{String, Int}()
    open(datapath, "r") do fin
        for line in eachline(fin)
            for word in split(rstrip(line))
                worddict[word] = get!(worddict, word, 0) + 1
            end
        end
    end

    open(joinpath(savedir, savefile), "w") do fdict
        # Add three special marks: <s> for start mark,
        # <e> for end mark and <p> for padding mark.
        foreach(x -> @printf(fdict, "%s\t-1\n", x), [BOS, EOS, PAD])
        foreach(x -> @printf(fdict, "%s\t%d\n", x[1], x[2]),
                sort(collect(worddict), by=x->x[2], rev=true))
    end
end

"""
    loadvocab(vocabfile)
"""
function loadvocab(vocabfile)
    @assert isfile(vocabfile) "$vocabfile does not exist."
    worddict = Dict{String, Int}()
    open(vocabfile, "r") do fdict
        for (index, line) in enumerate(eachline(fdict))
            word, frequency = split(line, "\t")
            get!(worddict, word, index)
        end
    end
    worddict
end

"""
    getvocab(savedir)

Return the word vocabulary for PTB dataset. The word vocabulary will be built
from PTB training data if it does not exist.
"""
function getvocab(savedir)
    vocabpath = joinpath(savedir, vocabfile)
    !isfile(vocabpath) && buildvocab(traindata, vocabpath, savedir)
    worddict = loadvocab(vocabpath)
end

"""
    getdatabatch(datafile; enable_shuffle=false)

Return data batches from the given datafile.
"""
function getdatabatch(batchsize, datafile, savedir; enable_shuffle=false)
    datapath = joinpath(savedir, datafile)
    isfile(datapath) || ptb_download(savedir)

    worddict = getvocab(savedir)
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

    xs = []
    ys = []
    for batch in Iterators.partition(dataset, batchsize)
        # Pad each batch to length of the longest sequence in the batch.
        maxlength = (length.(batch) |> maximum) - 1
        xs_ = batchseq([sample[1 : end - 1] for sample in batch], padid, maxlength)
        ys_ = batchseq([sample[2 : end] for sample in batch], padid, maxlength)

        push!(xs, xs_)
        push!(ys, ys_)
    end
    zip(xs, ys)
end

function getvocabdim(savedir)
    vocabpath = joinpath(savedir, vocabfile)
    !isfile(vocabpath) && buildvocab(traindata, vocabpath, savedir)
    return length(readlines(open(vocabpath)))
end

trainbatch(batchsize::Int; savedir) = getdatabatch(
    batchsize, traindata, savedir; enable_shuffle=true)
devbatch(batchsize::Int; savedir) = getdatabatch(
    batchsize, validdata, savedir; enable_shuffle=false)
testbatch(batchsize::Int; savedir) = getdatabatch(
    batchsize, testdata; enable_shuffle=false)

end
