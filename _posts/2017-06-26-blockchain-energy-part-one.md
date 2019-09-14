---
title: Blockchain in Energy 
date: 2017-06-26
categories:
  - Energy
excerpt: An introduction to blockchain the technology.

---

This post will give an introduction to the blockchain technology - what it is, how it works, the advantages and the challenges.   We will also take a look at two of the largest implementations of blockchain, bitcoin and Ethereum.

## What is blockchain

Blockchain is a technology that **enables decentralization**. It allows a true peer to peer economy - no third party needed. Blockchain can offer progress in three dimensions - **security, access and efficiency**.  Blockchain aligns with our transition towards a decentralized and clean energy future.

![Figure 1 - Our current energy transition is moving us away from dispatchable, centralized and large-scale generation towards intermittent, distributed and small scale generation.]({{"/assets/blockchain/fig1.png"}})

## The Times They Are a-Chanin'

Large scale third parties provide the trust in today's transaction systems and markets. They perform vital market roles such as identification, clearing, settling and keeping records.

But large scale third parties have downsides that blockchains can address. Blockchain offers advantages in three dimensions - security, access and efficiency.

A centralized system offers a single point of failure for attack - reducing security. Third parties can limit access to markets and data - reducing access. They introduce temporal & monetary costs by increasing the complexity of transactions - reducing efficiency.

Blockchain stores data on every member node - **increasing security**. Public blockchains are open to all, with individuals able to transact on an equal footing with large corporations - **increasing access**. Peer to peer transactions are made quickly and at low cost - **improving efficiency**.

Blockchain can also enable new kinds of transactions and business models. There are exciting applications in the energy industry beyond managing financial transactions - the future of these is not clear.  Currently the energy and blockchain are at the pilot stages, unproven on large scales.

There are significant technical, legal and regulatory challenges to overcome. Yet these challenges could be far smaller than the potential benefit blockchains could bring to our distributed and clean energy future.

## Queen Chain Approximately

Two key technical innovations in blockchain are the **decentralization of data storage and verification**. They both contribute towards the increased security, access and efficiency of the blockchain.

Decentralization of data storage improves security and access. Each member stores a copy of the entire blockchain. This removes the possibility losing data from a central database - improving security. 

As each member stores the entire blockchain, she can access infomation about any transaction - democratizing data access. A public blockchain is open for all to join - democratizing market access.

Decentralization of verification improves access and efficiency. By allowing any node to verify transactions, no central authority can limit access to the market - improving access. Rewarding the fastest verification incentives reduced transaction time and cost - improving efficiency.

Verification of decentralization is what allows peer to peer transactions. The ability of the collective to verify transactions means you don't need to wait for a central authority to authorize your transaction.

So decentralization of data and verification leads to improved security, access and efficiency. But what actually is allowing these improvements?

**The magic of blockchain is how it maintains the true blockchain state**. Blockchain truth is measured by the amount of a scarce resource attached to that blockchain. The scarce resource is supplied by blockchain members.

By making the correct blockchain require resource behind it, it makes proof of the blockchain rely on something that can't be faked. This is the magic of the blockchain - making digital infomation behave like a scarce physical asset.

Falsifying the blockchain requires attaching more resource to a competitor blockchain. This means a blockchain is not immutable. A transaction can be modified by altering all subsequent blocks with the collusion of the majority of the blockchain (a 51% attack).

This fake blockchain could outcompete the true blockchain if enough resource was put behind it. However, once a true blockchain gets far enough ahead this task becomes very difficult.

## Jenny From The Block

A key challenge in blockchain is what mechanism to use for validating transactions.  A simplifed description of the mechanics of **proof of work** and **proof of stake** blockchains are given below.  Currently there is no clear consensus on which mechanism is optimal - so there is significant innovation occurring in developing new mechanisms.

### Proof of work

Private and public keys ensure the security of each member. A private key can access an account and execute a transaction. A public key can view the transaction history of the blockchain.

Although this public key is visible to all the identity of the member is not revealed. In this way a blockchain paradoxically provides both complete anonymity but no privacy about transaction history.

Blockchain members (known as nodes) run the blockchain. Each node can generate & digitally sign transactions using their private key. Transactions are added to a list held locally by the node and rapidly forwarded onto other nodes.

![[Figure 2 - the blockchain process](https://www.pwc.ch/en/2017/pdf/pwc_blockchain_opportunity_for_energy_producers_and_consumers_en.pdf)]({{"/assets/blockchain/fig2.png"}})

It's then the role of miners' (verification nodes) to create blocks from these lists of transactions. Miners first check the balances of both parties - available through the distributed blockchain. Miners then compete to solve a mathematical problem that allows transactions to be chained together. Solving the problem requires scarce real world resources (hardware and energy) meaning there is no way to cheat.
  
![[Figure 3 - the verification process](https://www.pwc.ch/en/2017/pdf/pwc_blockchain_opportunity_for_energy_producers_and_consumers_en.pdf)]({{"/assets/blockchain/fig3.png"}})

Solving this problem is not computing anything useful - but requires computational resource to be solved correctly. **This is how proof of work attaches a scarce resource (hardware and energy) to digital infomation (the true blockchain).**

After the new block is added to the blockchain, it is propagated around the network in a similar fashion to transactions. Any conflicting transactions on nodes are discarded. It makes sense for other miners to add that validated block to their own blockchain and to work on the next block. This alignment of incentives allows the true blockchain to outgrow the others as more and more miners get rewarded to extend it.

The disadvantage of the proof of work consensus process is that all this computation requires a lot of electricity. This cost is also paid even if no one is trying to disrupt the blockchain.

In 2015 the electricity consumed in a single Bitcoin transaction was enough to power 1.57 American households for one day. If we assume an annual average consumption of `10.8 MWh` and an electricity cost of `£30/MWh`, this equates to a cost of `£1.60` per transaction. There is also the carbon impact of this electricity consumption.

### Proof of stake

**Proof of stake is a solution to the electricity consumption problem of proof of work**. In a proof of stake system validators must place a deposit of cryptocurrency in their block. Blockchain truth is deterministically determined by the amount of currency attached.

In both systems the influence of a user depends on the amount of scarce resource they put behind a blockchain. In a proof of work system the resource is computational; in a proof of stake system the resource is financial.

## Smart contracts

The ability to deploy smart contracts is one of the blockchains key strengths. It's how we can infuse a blockchain system with intelligence and movement. It enables new types of transactions, business models and markets on the blockchain.

A smart contract is an application that runs on the blockchain. Smart contracts improve the efficiency of the blockchain by automating manual processes. Smart contracts can allow new kinds of business models on the blockchain.

**Smart contracts allow innovation to occur on a small scale**. No central third party can limit the experimentation of smart contracts. A blockchain can support diverse ecosystem of smart contracts.

## Bitcoin and Ethereum

Today the most largest and most visible use of the blockchain technology are in cryptocurrencies like bitcoin and Ethereum.

Bitcoin is both a cryptocurrency and a digital payment system. The main blockchain of bitcoin is public - anyone can mine, buy and trade bitcoin. The technology behind bitcoin is open source. Yet bitcoin is not an ideal blockchain implementation - **on key KPIs such as throughput speed, latency bitcoin lags well behind VISA.**

Storage of the blockchain is also an issue - in April 2017 the size of the bitcoin ledger was around 200 GB. If VISA were implemented using a blockchain it would take up around 200 PB per year. Electricity consumption is also a problem - **in April 2017 bitcoin mining consumed around 300 MW of electricity.**

Ethereum is a bitcoin competitor. Ethereum is also a public, open source blockchain with it's own currency known as 'ether'.  **The key advantage of Ethereum over bitcoin is that it allows more flexible smart contracts**. Ethereum has an integrated, Turing complete and all purpose programming language. Bitcoin's language is not Turing complete - meaning there are solvable problems that bitcoin's language cannot solve.

Bitoins's smart contracts limit it to being a currency and payment system. Ethererum could allow many different kinds of assets to be traded for many different business purposes.

Another difference between bitcoin and Ethereum is the algorithm used for reaching a consensus on the true state of the blockchain. Both historically use proof of work to reach a consensus on blockchain truth. Recently Ethereum is phasing in proof of stake as a lower cost way of reaching consensus on blockchain truth.

Thanks for reading!

## Sources and further reading

### White papers

  * [Bitcoin: A Peer-to-Peer Electronic Cash System](https://bitcoin.org/bitcoin.pdf)
  * [A Next-Generation Smart Contract and Decentralized Application Platform (Ethereum)](https://github.com/ethereum/wiki/wiki/White-Paper)

### Talks & Podcasts

  * [Blockchain Revolution - Alex Tapscott](https://www.youtube.com/watch?v=3PdO7zVqOwc)
  * [The Blockchain: Enabling a Distributed and Connected Energy Future](https://www.youtube.com/watch?v=cpMwPhA9QzM)
  * [Blockchain is Eating Wall Street - Alex Tapscott](https://www.youtube.com/watch?v=WnEYakUxsHU)
  * [The Quiet Master of Cryptocurrency — Nick Szabo (Tim Ferriss)](http://tim.blog/2017/06/04/nick-szabo/)
  * [What is Blockchain](https://www.youtube.com/watch?v=93E_GzvpMA0)

### Reports & Articles

  * [Blockchain - an opportunity for energy producers and consumers?](https://www.pwc.com/gx/en/industries/assets/pwc-blockchain-opportunity-for-energy-producers-and-consumers.pdf)
  * [Blockchain applications in energy trading](https://www2.deloitte.com/content/dam/Deloitte/uk/Documents/energy-resources/deloitte-uk-blockchain-applications-in-energy-trading.pdf)
  * [How blockchain could upend power markets](http://www.energycentral.com/c/iu/how-blockchain-could-upend-power-markets)
  *  [Enerchain: A Decentralized Market on the Blockchain for Energy Wholesalers](http://spectrum.ieee.org/energywise/energy/the-smarter-grid/enerchain-a-decentralized-market-on-the-blockchain-for-energy-wholesalers)
  * [What does blockchain mean for energy? - Electron](http://www.electron.org.uk/blog.html)
  * [How Utilities Are Using Blockchain to Modernize the Grid](https://hbr.org/2017/03/how-utilities-are-using-blockchain-to-modernize-the-grid)
  * [The Energy Blockchain: How Bitcoin Could Be a Catalyst for the Distributed Grid](https://www.greentechmedia.com/articles/read/the-energy-blockchain-could-bitcoin-be-a-catalyst-for-the-distributed-grid)
  * [Proof of Work vs Proof of Stake: Basic Mining Guide](https://blockgeeks.com/guides/proof-of-work-vs-proof-of-stake/)
  *  [What is Proof of Stake](https://github.com/ethereum/wiki/wiki/Proof-of-Stake-FAQ) 
  *  [Why Do I Need a Public and Private Key on the Blockchain?](https://blog.wetrust.io/why-do-i-need-a-public-and-private-key-on-the-blockchain-c2ea74a69e76)
  * [What are Bitcoin miners actually solving?](https://www.quora.com/What-are-Bitcoin-miners-actually-solving-What-kind-of-math-problems-are-they-solving-and-what-do-they-achieve-by-solving-them)
  * [The Blockchain Immutability Myth](http://www.coindesk.com/blockchain-immutability-myth/)
  * [Are Off-Block Chain Transactions Bad for Bitcoin?](https://www.coindesk.com/block-chain-transactions-bad-bitcoin/)
