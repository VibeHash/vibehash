# Glossary

Short definitions of the terms used across these docs, in alphabetical order.

**Adapter.** The small set of trained weights a fine-tune produces, loaded on top of its base model with Hugging Face transformers, PEFT or vLLM.

**Base model.** The pretrained open model a fine-tune starts from, such as a Qwen model from Hugging Face. Its weights stay frozen; only the adapter is trained.

**Beacon.** A round's random number, made after the commit deadline from the previous block, the leader's committed secret and every commitment. It picks which segments get checked.

**Block.** One round's result on the chain: its transactions, its round summary, the hash of its round record and of the block before it, signed by its proposer.

**Bootstrap stake.** Stake a genesis gives a network's first validators so they can vote before anyone has earned anything. It can never be withdrawn or spent, and is burned if its validator leaves.

**Chain id.** A network's name, written into every block and every signature, so a message signed for one network is never valid on another.

**Checker.** A staked worker chosen at random to re-run parts of another worker's unit and report whether they match. Checking is paid work. Also called a verifier.

**Commit certificate.** The validators' signed precommits that made a block final. Anyone can check it without trusting the node that served the block.

**Commitment.** A worker's sealed fingerprint of its result, sent before anything is revealed, so nobody can copy an answer or change theirs later.

**DeepSeek prep.** The first stage of a DeepSeek job: running the model's frozen bottom layers over the job's text once, so miners then train only the top layers. The prep is itself mining work, in units miners run and check.

**Dispute.** A fine-tune's disagreement between a worker and a checker, settled by narrowing it down to one small calculation that the node referees.

**Double-signing.** A validator signing two conflicting votes or proposals at the same height. It costs all of the validator's stake and bans it from the set for good.

**Earn-in.** A network rule letting workers start with no stake: their first 5 VIBE of pay becomes their stake.

**Escrow.** A job's budget, locked when it's posted and paid out round by round for accepted work. Cancelling returns what's left.

**Evo (Evo Hash).** Training by evolution strategies. Miners answer questions with slightly altered copies of the model, and blocks keep only the seeds and scores, so the trained model can be worked out from the chain.

**Follower.** A node with no vote. It syncs and checks the chain, serves it, passes transactions on, and points workers at the current leader.

**Genesis.** The file a network starts from: chain id, starting balances and stakes, validators and issuance schedule. Its hash identifies the network.

**Halt.** A node stopping itself because a final block couldn't be saved to disk. A restart carries on from what it saved.

**Halving.** The point at which the block subsidy halves. On current networks halvings count rounds that paid for work.

**Job.** A training task someone posts with a budget: a model to train, its data and its settings. Workers train it round by round.

**Leader.** The validator running the current round and proposing its block. Validators take turns, in a stake-weighted order.

**LoRA.** Low-rank adaptation: fine-tuning a large model by training small extra matrices beside some layers instead of all its weights. What fine-tune jobs train.

**Phase.** One part of a round: register, compute, reveal and verify. By default they take 3, 20, 5 and 10 seconds.

**Proposer.** The validator that proposes a block. It gets 5% of the round's pool if the round verified work, plus the block's transaction fees.

**Replay.** Re-running a segment of training from a worker's checkpoint and comparing the result, within tolerance.

**Reputation.** A multiplier between 0.5 and 1.0 on a worker's pay, earned round by round.

**Round.** One cycle of work, about 38 seconds by default, ending in a block.

**Round record.** Every signed message a round was decided from. A block commits to it, so anyone can replay the round.

**Segment.** One of the 8 pieces a unit is split into, with a checkpoint at each end.

**Slashing.** Taking part of a cheater's stake: half for a failed check, all of it for a validator that double-signs.

**Stake.** VIBE locked as a bond. It backs a worker's checks and a validator's votes, and it's what a cheat loses.

**Subsidy.** The new VIBE a round mints.

**Syn.** The smallest unit of VIBE: 1 VIBE = 100,000,000 syn.

**Tolerance.** How far a replay may differ from a worker's checkpoint and still pass, allowing for rounding differences between chips.

**Unbonding.** The 20 rounds unstaked coins wait before they can be spent, during which they can still be slashed.

**Unit.** A small piece of training work: a number of steps from this round's model with batches from a given seed.

**Validator.** A staked node that votes on blocks and takes turns leading rounds.

**Worker.** A computer that mines by training units and checking other workers' units. Also called a miner.
