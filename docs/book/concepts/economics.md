# Economics

Where VIBE comes from, how each round's pay is split, and what stake is for. Each network fixes its own issuance schedule in its genesis.

{% hint style="warning" %}
Vibe Hash is a testnet. VIBE has no value, and the numbers on this page may change before any mainnet.
{% endhint %}

All amounts are kept as whole numbers of **syn**: 1 VIBE = 100,000,000 syn. Rounding leftovers are burned, so the supply always adds up exactly.

## The pool: subsidy plus fees

Each round has a pool of VIBE to pay out:

- **The block subsidy**: new coins, minted by the round.
- **The job's fee release**: the job poster's payment for the units that passed its checks. Posters pay nothing for work that failed.

A job poster puts a budget in **escrow** when posting a job, and each round releases the fee for that round's accepted units from it. Cancelling a job returns what's left.

## Issuance schedules

| | testnet | mainnet |
|---|---|---|
| Subsidy per round that pays for work | 50 VIBE | 3.125 VIBE |
| Halving interval | 210 rounds that paid for work | 3,320,000 rounds that paid for work |
| One halving, if every 38 s round is worked | about 2.2 hours | about 4 years |
| Issued in total | about 21,000 VIBE | about 20,750,000 VIBE |
| Cap on all coins ever (genesis included) | none | 21,000,000 VIBE |

On networks made with current releases, **only a round that pays for work mints**, and the halvings count those rounds. A round in which nobody's work is accepted mints nothing, so a quiet network delays its halvings rather than losing coins. The mainnet's cap is enforced: a genesis that allocates more is invalid, and a subsidy never takes supply past it.

Older networks keep the original rule: every block that seals a round mints its subsidy, whatever nobody earned is burned, and halvings count block heights.

## How a round's pay is split

| Share | Who | Split by |
|---|---|---|
| 80% | Workers | verified training work × reputation |
| 15% | Checkers | re-run work × reputation |
| 5% | Block proposer | paid only if the round verified some work |

Anything nobody earned, such as the checkers' share in a round nobody checked, is burned rather than handed to the proposer, so nobody gains by rejecting good work. Pay per unit of work is the same on every machine: a laptop earns less in total, but gets paid every round it contributes.

**Reputation** scales a share between 0.5 and 1.0. New wallets start at 0.8, gain 0.02 for each round with verified work, lose 0.05 for a missed deadline, and drop to 0.5 on fraud. It moves per round, not per unit, so a small machine builds it as fast as a big one.

## Transaction fees

Every transaction pays a small fee to the validator that proposes its block. Nodes relay transactions paying at least their minimum, 0.0001 VIBE by default. Fees move coins rather than mint them. The fee comes out of spendable balance, so keep a little unstaked VIBE to unstake with. Workers' messages are free.

## Stake

Stake is what a cheater stands to lose.

- **Workers** need 5 VIBE staked. On a network with **earn-in** (the default for new networks), a new worker starts with nothing: its first 5 VIBE of pay goes into its stake. Until then it gets small batches of work, after staked workers, and never checks others' results.
- **Validators** need 100 VIBE staked to join the set, and vote with all of it.

## Unbonding

Unstaked coins wait in an **unbonding** queue for 20 rounds before they can be spent, and can still be slashed during that time. So a cheater can't pull their stake out before the verdict lands.

## Bootstrap stake

When a network starts, nobody has earned anything yet, so the genesis can give its first validators **bootstrap stake**. It counts for voting and proposing, but it can never be withdrawn or spent, and it's burned if its validator leaves. If it's slashed, that part is always burned, never paid to anyone. As others earn stake and join, it gets outweighed. So apart from any coins a genesis hands out to pay for the first jobs, every spendable coin starts from work.

## Penalties

| What happened | Stake | Pay |
|---|---|---|
| Missed the compute deadline | kept | that unit unpaid |
| Crashed after sending a sealed answer | kept | whole round unpaid |
| Failed a check, approved a planted fake, or rejected honest work | 50% slashed | whole round unpaid |
| A validator signed two conflicting votes | all of it, and banned | — |

Half of a 50% slash goes to whoever caught it and half is burned. For a double-signing validator, the reporter gets a tenth of the withdrawable part and the rest is burned.

See [How the network checks work](verification.md) for how cheats are caught, and [VIBE: balance, sending and staking](../wallet/vibe.md) for doing it yourself.
