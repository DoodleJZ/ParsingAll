import argparse
import itertools
import os.path
import time
import uuid

import torch
import torch.optim.lr_scheduler

import numpy as np
import math
import json
from Evaluator import evaluate
from Evaluator import dep_eval
from Evaluator import srl_eval
from Evaluator import pos_eval
from Datareader import syndep_reader
from Datareader import srlspan_reader
from Datareader import srldep_reader
import trees
import vocabulary
import makehp
import Zparser
import utils

tokens = Zparser

uid = uuid.uuid4().hex[:6]

def torch_load(load_path):
    if Zparser.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return makehp.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=300,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,

        partitioned=True,
        use_cat=False,
        use_lstm = False,
        joint_syn_dep = False,
        joint_syn_const = False,
        joint_srl_dep = False,
        joint_srl_span = False,
        joint_pos = False,

        use_gold_predicate = False,
        use_bispan_respresent = False,
        use_syncatspan = False,
        use_addspan = False,
        use_catspan = False,
        use_srl_biaffine = False,
        use_srl_dot = False,
        use_srl_jointdecode = False,

        const_lada = 0.5,
        labmda_verb = 0.6,
        labmda_span = 0.6,
        max_num_span = 300,
        max_num_verb = 30,
        use_span_ff = False,
        use_prespan_ff = False,
        use_verb_ff = False,
        use_softmax_verb = False,
        use_softmax_span = False,
        use_softmax_srlabel = True,

        num_layers=12,
        d_model=1024,
        num_heads=8,#12
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_biaffine = 1024,
        d_score_hidden = 256,
        d_verb = 512,
        d_span = 512,
        d_prespan = 512,

        attention_dropout=0.2,
        embedding_dropout=0.2,
        relu_dropout=0.2,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_elmo = False,
        use_bert=False,
        use_chars_lstm=False,

        dataset = 'ptb',

        model_name = "dep+const",
        embedding_type = 'random',
        #['glove','sskip','random']
        embedding_path = "/data/glove.gz",
        punctuation='.' '``' "''" ':' ',',

        d_char_emb = 64, # A larger value may be better for use_chars_lstm

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-87-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        )

def count_wh(str, data):
    cun_w = 0
    for i, c_tree in enumerate(data):
        nodes = [c_tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                cun_w += node.cun_w
                nodes.extend(reversed(node.children))

    print("total wrong head of :", str, "is", cun_w)

def align_sent(true_sents, wrong_sents, align_path):
    if not os.path.exists(align_path):
        align_dict = {}
        for i, t_sents in enumerate(true_sents):
            flag = 0
            for j, w_sents in enumerate(wrong_sents):
                #print(w_sents, t_sents)
                if w_sents == t_sents:
                    align_dict[i] = j
                    flag = 1
                    break
            if flag == 0:
                align_dict[i] = -1
            if j % 5000 == 0:
                print("done aligning", j)
        json.dump(align_dict, open(align_path, 'w'))
    else:
        with open(align_path, 'r') as f:
            align_dict = json.load(fp=f)

    return align_dict

def make_align(align_dict, sent_w, verb_w, dict_w):
    sent = []
    verb = []
    dict = []
    for cun, i in align_dict.items():
        if i != -1:
            sent.append(sent_w[i])
            verb.append(verb_w[i])
            dict.append(dict_w[i])
        else:
            sent.append(None)
            verb.append(None)
            dict.append(None)

    return  sent, verb, dict

def correct_sent(syndep_sents, srlspan_sents, srldep_sents):
    for i, (syndep_sent, srlspan_sent, srldep_sent) in enumerate(zip(syndep_sents, srlspan_sents, srldep_sents)):

        assert len(syndep_sent) == len(srlspan_sent)
        if srldep_sent is not None:
            assert len(syndep_sent) == len(srldep_sent)

def span_miss_verb(srlspan_verb, srldep_verb):
    cun = 0
    for span_verb, dep_verb in zip(srlspan_verb, srldep_verb):
        dep_verb_list = [verb[0] for verb in dep_verb]
        for verb in span_verb:
            if verb not in dep_verb_list:
                cun += 1
    print("span miss verb ", cun)

def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)
    print("Hyperparameters:")
    hparams.print()

    #srl dev set which uses 24 section of ptb is different from syn
    synconst_train_path = args.synconst_train_ptb_path
    synconst_dev_path = args.synconst_dev_ptb_path

    syndep_train_path = args.syndep_train_ptb_path
    syndep_dev_path = args.syndep_dev_ptb_path

    srlspan_train_path = args.srlspan_train_ptb_path
    srlspan_dev_path = args.srlspan_dev_ptb_path

    srldep_train_path = args.srldep_train_ptb_path
    srldep_dev_path = args.srldep_dev_ptb_path

    seldep_train_align_path = "data/seldep_train_align_path.json"

    syndep_train_sent, syndep_train_pos, syndep_train_heads, syndep_train_types = syndep_reader.read_syndep(
        syndep_train_path, hparams.max_len_train)

    syndep_dev_sent, syndep_dev_pos, syndep_dev_heads, syndep_dev_types = syndep_reader.read_syndep(
        syndep_dev_path, hparams.max_len_dev)

    srlspan_train_sent, srlspan_train_verb, srlspan_train_dict, srlspan_train_predpos, srlspan_train_goldpos, \
    srlspan_train_label, srlspan_train_label_start, srlspan_train_heads = srlspan_reader.read_srlspan(srlspan_train_path, hparams.max_len_train)

    srlspan_dev_sent, srlspan_dev_verb, srlspan_dev_dict, srlspan_dev_predpos, srlspan_dev_goldpos, \
    srlspan_dev_label, srlspan_dev_label_start, srlspan_dev_heads = srlspan_reader.read_srlspan(srlspan_dev_path, hparams.max_len_dev)

    srldep_train_sent_w, srldep_train_predpos_w, srldep_train_verb_w, srldep_train_dict_w, srldep_train_heads_w = srldep_reader.read_srldep(srldep_train_path,hparams.max_len_train)
    srldep_dev_sent, srldep_dev_predpos, srldep_dev_verb, srldep_dev_dict, srldep_dev_heads = srldep_reader.read_srldep(srldep_dev_path,hparams.max_len_dev)

    print("aligning srl dep...")
    srldep_train_align_dict = align_sent(srlspan_train_sent, srldep_train_sent_w, seldep_train_align_path)
    srldep_train_sent ,srldep_train_verb ,srldep_train_dict  = make_align(srldep_train_align_dict, srldep_train_sent_w,
                                                                          srldep_train_verb_w, srldep_train_dict_w)
    print("correct sents...")
    correct_sent(syndep_train_sent, srlspan_train_sent, srldep_train_sent)


    print("Loading training trees from {}...".format(synconst_train_path))
    train_treebank = trees.load_trees(synconst_train_path, syndep_train_heads, syndep_train_types, srlspan_train_goldpos)
    if hparams.max_len_train > 0:
        train_treebank = [tree for tree in train_treebank if len(list(tree.leaves())) <= hparams.max_len_train]
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(synconst_dev_path))
    dev_treebank = trees.load_trees(synconst_dev_path, syndep_dev_heads, syndep_dev_types, None, None)
    #different dev, srl is empty
    if hparams.max_len_dev > 0:
        dev_treebank = [tree for tree in dev_treebank if len(list(tree.leaves())) <= hparams.max_len_dev]
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]
    dev_parse = [tree.convert() for tree in dev_treebank]

    count_wh("train data:", train_parse)
    count_wh("dev data:", dev_parse)

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(Zparser.START)
    tag_vocab.index(Zparser.STOP)
    tag_vocab.index(Zparser.TAG_UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(Zparser.START)
    word_vocab.index(Zparser.STOP)
    word_vocab.index(Zparser.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())
    sublabels = [Zparser.Sub_Head]
    label_vocab.index(tuple(sublabels))

    type_vocab = vocabulary.Vocabulary()

    srl_vocab = vocabulary.Vocabulary()
    srl_vocab.index('*')

    for srl_dict in srldep_train_dict:
        if srl_dict is not None:
            for verb_id, arg_list in srl_dict.items():
                for arg in arg_list:
                    srl_vocab.index(arg[1])

    for srl_dict in srlspan_train_dict:
        if srl_dict is not None:
            for verb_id, arg_list in srl_dict.items():
                for arg in arg_list:
                    srl_vocab.index(arg[2])

    char_set = set()

    for i, tree in enumerate(train_parse):

        const_sentences = [leaf.word for leaf in tree.leaves()]
        assert len(const_sentences)  == len(syndep_train_sent[i])
        assert len(const_sentences) == len(srlspan_train_sent[i])
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                type_vocab.index(node.type)
                char_set |= set(node.word)

    char_vocab = vocabulary.Vocabulary()

    #char_vocab.index(tokens.CHAR_PAD)

    # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        # This also takes care of constants like tokens.CHAR_PAD
        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index(tokens.CHAR_UNK)
        char_vocab.index(tokens.CHAR_START_SENTENCE)
        char_vocab.index(tokens.CHAR_START_WORD)
        char_vocab.index(tokens.CHAR_STOP_WORD)
        char_vocab.index(tokens.CHAR_STOP_SENTENCE)
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()
    type_vocab.freeze()
    srl_vocab.freeze()

    punct_set = hparams.punctuation

    def print_vocabulary(name, vocab):
        special = {tokens.START, tokens.STOP, tokens.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)
        print_vocabulary("Char", char_vocab)
        print_vocabulary("Type", type_vocab)
        print_vocabulary("Srl", srl_vocab)


    print("Initializing model...")

    load_path = None
    if load_path is not None:
        print(f"Loading parameters from {load_path}")
        info = torch_load(load_path)
        parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])
    else:
        parser = Zparser.ChartParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            type_vocab,
            srl_vocab,
            hparams,
        )

    print("Initializing optimizer...")
    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    if load_path is not None:
        trainer.load_state_dict(info['trainer'])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_score = -np.inf
    best_model_path = None
    model_path = None
    model_name = hparams.model_name

    print("This is ", model_name)
    start_time = time.time()

    best_epoch = 0
    def check_dev(epoch_num):
        nonlocal best_dev_score
        nonlocal best_model_path
        nonlocal best_epoch

        print("Start dev eval:")

        dev_start_time = time.time()
        dev_fscore = evaluate.FScore(0, 0, 0)
        dev_uas = 0
        dev_las = 0
        pos_dev = 0
        summary_dict = {}
        summary_dict["srlspan dev F1"]= 0
        summary_dict["srldep dev F1"] = 0
        parser.eval()

        syntree_pred = []
        srlspan_pred = []
        srldep_pred = []
        pos_pred = []
        if hparams.joint_syn_dep or hparams.joint_syn_const:
            for dev_start_index in range(0, len(dev_treebank), args.eval_batch_size):
                subbatch_trees = dev_treebank[dev_start_index:dev_start_index+args.eval_batch_size]
                subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]

                syntree, _, _= parser.parse_batch(subbatch_sentences)

                syntree_pred.extend(syntree)

            #const parsing:

            dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, syntree_pred)

            #dep parsing:

            dev_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
            dev_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
            assert len(dev_pred_head) == len(dev_pred_type)
            assert len(dev_pred_type) == len(syndep_dev_types)
            dev_uas, dev_las = dep_eval.eval(len(dev_pred_head), syndep_dev_sent, syndep_dev_pos,
                                                                      dev_pred_head, dev_pred_type,
                                                                      syndep_dev_heads, syndep_dev_types,
                                                                      punct_set=punct_set,
                                                                      symbolic_root=False)
        #for srl different dev set
        if hparams.joint_srl_span or hparams.joint_pos:
            for dev_start_index in range(0, len(srlspan_dev_sent), args.eval_batch_size):
                subbatch_words = srlspan_dev_sent[dev_start_index:dev_start_index+args.eval_batch_size]
                subbatch_pos = srlspan_dev_predpos[dev_start_index:dev_start_index + args.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i,(tags, words) in enumerate(zip(subbatch_pos, subbatch_words))]

                if hparams.use_gold_predicate:
                    srlspan_tree, srlspan_dict, _ = parser.parse_batch(subbatch_sentences, gold_verbs=srlspan_dev_verb[dev_start_index:dev_start_index + args.eval_batch_size],
                                                            syndep_heads = srlspan_dev_heads[dev_start_index:dev_start_index + args.eval_batch_size])
                else:
                    srlspan_tree, srlspan_dict, _= parser.parse_batch(subbatch_sentences, syndep_heads = srlspan_dev_heads[dev_start_index:dev_start_index + args.eval_batch_size])

                srlspan_pred.extend(srlspan_dict)

                pos_pred.extend([[leaf.goldtag for leaf in tree.leaves()] for tree in srlspan_tree])

            if hparams.joint_srl_span:
                print("===============================================")
                print("srl span dev eval:")
                precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
                    srl_eval.compute_srl_f1(srlspan_dev_sent, srlspan_dev_dict, srlspan_pred,
                                            srl_conll_eval_path=False))
                summary_dict["srlspan dev F1"] = f1
                summary_dict["srlspan dev precision"] = precision
                summary_dict["srlspan dev recall"] = precision

            if hparams.joint_pos:
                pos_dev = pos_eval.eval(srlspan_dev_goldpos, pos_pred)

        if hparams.joint_srl_dep:

            for dev_start_index in range(0, len(srldep_dev_sent), args.eval_batch_size):
                subbatch_words = srldep_dev_sent[dev_start_index:dev_start_index+args.eval_batch_size]
                subbatch_pos = srldep_dev_predpos[dev_start_index:dev_start_index + args.eval_batch_size]
                subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i,(tags, words) in enumerate(zip(subbatch_pos, subbatch_words))]

                if hparams.use_gold_predicate:
                    _, _, srldep_dict= parser.parse_batch(subbatch_sentences, gold_verbs=srldep_dev_verb[dev_start_index:dev_start_index + args.eval_batch_size]
                                                           , syndep_heads=srldep_dev_heads[
                                                                          dev_start_index:dev_start_index + args.eval_batch_size])
                else:
                    _, _, srldep_dict= parser.parse_batch(subbatch_sentences, syndep_heads = srldep_dev_heads[dev_start_index:dev_start_index + args.eval_batch_size])

                srldep_pred.extend(srldep_dict)

            print("===============================================")
            print("srl dep dev eval:")
            precision, recall, f1 = (
                srl_eval.compute_dependency_f1(srldep_dev_sent, srldep_dev_dict, srldep_pred,
                                               srl_conll_eval_path=False, use_gold = hparams.use_gold_predicate))
            summary_dict["srldep dev F1"] = f1
            summary_dict["srldep dev precision"] = precision
            summary_dict["srldep dev recall"] = precision
            print("===============================================")

        print(
            "dev-elapsed {} "
            "total-elapsed {}".format(
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        print(
            '============================================================================================================================')

        if dev_fscore.fscore + dev_las + summary_dict["srlspan dev F1"] + summary_dict["srldep dev F1"] + pos_dev > best_dev_score :
            if best_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_score = dev_fscore.fscore + dev_las + summary_dict["srlspan dev F1"] + summary_dict["srldep dev F1"] + pos_dev
            best_model_path = "{}_best_dev={:.2f}_devuas={:.2f}_devlas={:.2f}_devsrlspan={:.2f}_devsrldep={:.2f}".format(
                args.model_path_base, dev_fscore.fscore, dev_uas,dev_las, summary_dict["srlspan dev F1"] , summary_dict["srldep dev F1"])
            print("Saving new best model to {}...".format(best_model_path))
            torch.save({
                'spec': parser.spec,
                'state_dict': parser.state_dict(),
                'trainer' : trainer.state_dict(),
                }, best_model_path + ".pt")


    # srlspan_train_dict = train_tree_srl

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break
        #check_dev(epoch)
        train_data = [(parse_tree, srlspan, srldep) for parse_tree, srlspan, srldep in zip(train_parse, srlspan_train_dict, srldep_train_dict)]
        np.random.shuffle(train_data)
        epoch_start_time = time.time()

        train_parse = [data[0] for data in train_data]
        srlspan_train_dict = [data[1] for data in train_data]
        srldep_train_dict = [data[2] for data in train_data]

        for start_index in range(0, len(train_parse), args.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // args.batch_size)

            parser.train()

            batch_loss_value = 0.0
            batch_loss_syndep = 0.0
            batch_loss_srl = 0.0
            batch_loss_synconst =0.0

            batch_trees = train_parse[start_index:start_index + args.batch_size]
            batch_srlspans = srlspan_train_dict[start_index:start_index + args.batch_size]
            batch_srldeps = srldep_train_dict[start_index:start_index + args.batch_size]
            batch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in batch_trees]

            for subbatch_sentences, subbatch_trees, subbatch_srlspans, subbatch_srldeps in parser.split_batch(batch_sentences, batch_trees, batch_srlspans, batch_srldeps, args.subbatch_max_tokens):
                loss, srl_loss, synconst_loss, syndep_loss = parser.parse_batch(subbatch_sentences, subbatch_trees, subbatch_srlspans, subbatch_srldeps, epoch)

                loss = loss / len(batch_trees)
                srl_total_loss = srl_loss / len(batch_trees)
                syndep_total_loss = syndep_loss / len(batch_trees)
                synconst_loss = synconst_loss / len(batch_trees)
                if loss > 0:
                    loss_value = float(loss.data.cpu().numpy())
                    batch_loss_value += loss_value

                if synconst_loss > 0:
                    batch_loss_synconst += float(synconst_loss.data.cpu().numpy())
                if syndep_total_loss > 0:
                    batch_loss_syndep += float(syndep_total_loss.data.cpu().numpy())
                if srl_total_loss > 0:
                    batch_loss_srl += float(srl_total_loss.data.cpu().numpy())

                if loss > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "srl-loss {:.4f} "
                "synconst-loss {:.4f} "
                "syndep-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    batch_loss_srl,
                    batch_loss_synconst,
                    batch_loss_syndep,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev(epoch)

        # adjust learning rate at the end of an epoch
        if hparams.step_decay:
            if (total_processed // args.batch_size + 1) > hparams.learning_rate_warmup_steps:
                scheduler.step(best_dev_score)


def run_test(args):


    synconst_test_path = args.synconst_test_ptb_path

    syndep_test_path = args.syndep_test_ptb_path

    srlspan_test_path = args.srlspan_test_ptb_path
    srlspan_brown_path = args.srlspan_test_brown_path

    srldep_test_path = args.srldep_test_ptb_path
    srldep_brown_path = args.srldep_test_brown_path

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = Zparser.ChartParser.from_spec(info['spec'], info['state_dict'])

    syndep_test_sent, syndep_test_pos, syndep_test_heads, syndep_test_types = syndep_reader.read_syndep(
        syndep_test_path)

    srlspan_test_sent, srlspan_test_verb, srlspan_test_dict, srlspan_test_predpos, srlspan_test_goldpos, \
    srlspan_test_label, srlspan_test_label_start, srlspan_test_heads = srlspan_reader.read_srlspan(srlspan_test_path)

    srlspan_brown_sent, srlspan_brown_verb, srlspan_brown_dict, srlspan_brown_predpos, srlspan_brown_goldpos, \
    srlspan_brown_label, srlspan_brown_label_start, srlspan_brown_heads = srlspan_reader.read_srlspan(srlspan_brown_path)

    srldep_test_sent, srldep_test_predpos, srldep_test_verb, srldep_test_dict, srldep_test_heads = srldep_reader.read_srldep(srldep_test_path)
    srldep_brown_sent, srldep_brown_predpos, srldep_brown_verb, srldep_brown_dict, srldep_brown_heads = srldep_reader.read_srldep(srldep_brown_path)

    print("Loading test trees from {}...".format(synconst_test_path))
    test_treebank = trees.load_trees(synconst_test_path, syndep_test_heads, syndep_test_types, srlspan_test_label,
                                     srlspan_test_label_start)

    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Parsing test sentences...")
    start_time = time.time()

    punct_set = '.' '``' "''" ':' ','

    parser.eval()
    print("Start test eval:")
    test_start_time = time.time()

    syntree_pred = []
    srlspan_pred = []
    srldep_pred = []
    #span srl and syn have same test data
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index:start_index + args.eval_batch_size]

        subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]


        if parser.hparams.use_gold_predicate:
            syntree, srlspan_dict, _ = parser.parse_batch(subbatch_sentences, gold_verbs=srlspan_test_verb[start_index:start_index + args.eval_batch_size])
        else:
            syntree, srlspan_dict, _ = parser.parse_batch(subbatch_sentences)

        syntree_pred.extend(syntree)
        srlspan_pred.extend(srlspan_dict)

    for start_index in range(0, len(srldep_test_sent), args.eval_batch_size):

        subbatch_words_srldep = srldep_test_sent[start_index:start_index + args.eval_batch_size]
        subbatch_pos_srldep = srldep_test_predpos[start_index:start_index + args.eval_batch_size]
        subbatch_sentences_srldep = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i, (tags, words)
                              in enumerate(zip(subbatch_pos_srldep, subbatch_words_srldep))]

        if parser.hparams.use_gold_predicate:
            _, _, srldep_dict = parser.parse_batch(subbatch_sentences_srldep, gold_verbs=srldep_test_verb[start_index:start_index + args.eval_batch_size])
        else:
            _, _, srldep_dict = parser.parse_batch(subbatch_sentences_srldep)

        srldep_pred.extend(srldep_dict)

    # const parsing:

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, syntree_pred)

    # dep parsing:

    test_pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
    test_pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]

    assert len(test_pred_head) == len(test_pred_type)
    assert len(test_pred_type) == len(syndep_test_types)
    test_uas, test_las = dep_eval.eval(len(test_pred_head), syndep_test_sent, syndep_test_pos, test_pred_head,
                                       test_pred_type, syndep_test_heads, syndep_test_types, punct_set=punct_set,
                                       symbolic_root=False)


    print("===============================================")
    print("wsj srl span test eval:")
    precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
        srl_eval.compute_srl_f1(srlspan_test_sent, srlspan_test_dict, srlspan_pred, srl_conll_eval_path=False))
    print("===============================================")
    print("wsj srl dep test eval:")
    precision, recall, f1 = (
        srl_eval.compute_dependency_f1(srldep_test_sent, srldep_test_dict, srldep_pred,
                                       srl_conll_eval_path=False))
    print("===============================================")

    print(
        '============================================================================================================================')

    syntree_pred = []
    srlspan_pred = []
    srldep_pred = []
    for start_index in range(0, len(srlspan_brown_sent), args.eval_batch_size):
        subbatch_words = srlspan_brown_sent[start_index:start_index + args.eval_batch_size]
        subbatch_pos = srlspan_brown_predpos[start_index:start_index + args.eval_batch_size]
        subbatch_sentences = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for i, (tags, words)
                              in enumerate(zip(subbatch_pos, subbatch_words))]

        if parser.hparams.use_gold_predicate:
            syntree, srlspan_dict, _ = parser.parse_batch(subbatch_sentences, gold_verbs=srlspan_brown_verb[start_index:start_index + args.eval_batch_size])
        else:
            syntree, srlspan_dict, _ = parser.parse_batch(subbatch_sentences)

        syntree_pred.extend(syntree)
        srlspan_pred.extend(srlspan_dict)

    for start_index in range(0, len(srldep_brown_sent), args.eval_batch_size):

        subbatch_words_srldep = srldep_brown_sent[start_index:start_index + args.eval_batch_size]
        subbatch_pos_srldep = srldep_brown_predpos[start_index:start_index + args.eval_batch_size]
        subbatch_sentences_srldep = [[(tag, word) for j, (tag, word) in enumerate(zip(tags, words))] for
                                     i, (tags, words) in enumerate(zip(subbatch_pos_srldep, subbatch_words_srldep))]

        if parser.hparams.use_gold_predicate:
            _, _, srldep_dict = parser.parse_batch(subbatch_sentences_srldep, gold_verbs=srldep_brown_verb[start_index:start_index + args.eval_batch_size])
        else:
            _, _, srldep_dict = parser.parse_batch(subbatch_sentences_srldep)

        srldep_pred.extend(srldep_dict)


    print("===============================================")
    print("brown srl span test eval:")
    precision, recall, f1, ul_prec, ul_recall, ul_f1 = (
        srl_eval.compute_srl_f1(srlspan_brown_sent, srlspan_brown_dict, srlspan_pred, srl_conll_eval_path=False))
    print("===============================================")
    print("brown srl dep test eval:")
    precision, recall, f1 = (
        srl_eval.compute_dependency_f1(srldep_brown_sent, srldep_brown_dict, srldep_pred,
                                       srl_conll_eval_path=False))
    print("===============================================")

    print(
        "test-elapsed {} "
        "total-elapsed {}".format(
            format_elapsed(test_start_time),
            format_elapsed(start_time),
        )
    )

    print(
        '============================================================================================================================')

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--embedding-path", required=True)
    subparser.add_argument("--embedding-type", default="random")

    subparser.add_argument("--model-name", default="dep+const")
    subparser.add_argument("--evalb-dir", default="EVALB/")

    subparser.add_argument("--dataset", default="ptb")

    subparser.add_argument("--synconst-train-ptb-path", default="data/02-21.10way.clean")
    subparser.add_argument("--synconst-dev-ptb-path", default="data/22.auto.clean")
    subparser.add_argument("--syndep-train-ptb-path", default="data/ptb_train_3.3.0.sd")
    subparser.add_argument("--syndep-dev-ptb-path", default="data/ptb_dev_3.3.0.sd")
    subparser.add_argument("--srlspan-train-ptb-path", default="data/srl_span_train.txt")
    subparser.add_argument("--srlspan-dev-ptb-path", default="data/srl_span_dev.txt")
    subparser.add_argument("--srldep-train-ptb-path", default="data/srl_dep_train.txt")
    subparser.add_argument("--srldep-dev-ptb-path", default="data/srl_dep_dev.txt")

    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=30)
    subparser.add_argument("--epochs", type=int, default=150)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--embedding-path", default="data/glove.gz")
    subparser.add_argument("--dataset", default="ptb")
    subparser.add_argument("--lamb", type=float, default=0.5)
    subparser.add_argument("--synconst-test-ptb-path", default="data/23.auto.clean")
    subparser.add_argument("--syndep-test-ptb-path", default="data/ptb_test_3.3.0.sd")
    subparser.add_argument("--srlspan-test-ptb-path", default="data/srl_span_testwsj.txt")
    subparser.add_argument("--srlspan-test-brown-path", default="data/srl_span_testbrown.txt")
    subparser.add_argument("--srldep-test-ptb-path", default="data/srl_dep_testwsj.txt")
    subparser.add_argument("--srldep-test-brown-path", default="data/srl_dep_testbrown.txt")
    subparser.add_argument("--eval-batch-size", type=int, default=30)

    args = parser.parse_args()
    args.callback(args)

# %%
if __name__ == "__main__":
    main()
