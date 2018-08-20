---
title: 'A blockchain built in Python'
date: 2018-08-13
excerpt:  Making use of some of Python's cool standard library features.

---

Implementation of blockchain in Python - [hosted on GitHub here](https://github.com/ADGEfficiency/blockchain).  I'm particularly proud of

- use of `defaultdict` and `namedtuple` from the standard library
- inheriting from list for the `Blockchain` and `Network` classes

Transactions and blocks are represented using `namedtuple`.  Both are immutable - use of `namedtuple` allows indexing by key and take care of stuff like `__repr__`.  They also represent the most primitive objects in our blockchain network.

- one network
- n nodes 
- c blockchains (one per node)
- b blocks (per chain)
- t transactions (per block)

For example, we could have a network with two nodes (miners or account holders).  Each of these holds a blockchain - the same blockchain if the network is currently in consensus.  Each blockchain is made of multiple blocks, with each block having a number of transactions.

```python
Transaction = namedtuple(
    'transaction',
    ['sender',
     'to',
     'amount',
     'signature'])

Block = namedtuple(
    'block',
    ['index',
     'timestamp',
     'proof',
     'previous_hash',
     'hash',
     'transactions'])
```

A blockchain is represented as a class, inheriting from list.  This gives us functionality like indexing and appending for free.  The elements of this list are blocks.  

```python
class BlockChain(list):

    def __init__(self):

        genesis_block = OrderedDict(
            {'index': 1,
             'timestamp': date.datetime.now(),
             'proof': 9,
             'previous_hash': 'hello world',
             'transactions': []}
        )

        self.append(
            Block(
                **genesis_block,
                hash=get_hash(genesis_block))
        )

    def update_transactions(self, new):
        return self[-1].transactions + new

    def next_block(self, new_transactions, proof):
        last_block = self[-1]
        next_block = OrderedDict(
            {'index': len(self) + 1,
             'timestamp': date.datetime.now(),
             'proof': proof,
             'previous_hash': last_block.hash,
             'transactions' : self.update_transactions(new_transactions)}
        )

        self.append(
            Block(**next_block, hash=get_hash(get_hash(next_block)))
        )
```

A node (i.e. miner or holder of tokens) is a simple class.  It holds infomation such as public and private keys, along with functionality to make, sign and verify transactions.  Each node holds it's own instance of the `BlockChain` class.

The final major object is the `Network`, implemented as a class that inherits from list.  The network is composed of multiple nodes.  The network performs the consensus algorithm (in this case checking for the longest chain).  It also validates transactions.

```python
class Network(list):
    """ should never have a BlockChain - only nodes have chains """

    def __init__(
            self,
            nodes,
            overdraft_limit
    ):
        list.__init__(self, nodes)
        self.proof = 9
        self.overdraft_limit = overdraft_limit

    def proof_of_work(self):
        self.proof = self.proof + 0.1
        sleep(self.proof)
        return self.proof

    def consensus(self):
        """ find the node that mined the block """
        chains = [node.chain for node in self]
        new_chain = chains[np.argmax([len(chain) for chain in chains])]

        for node in self:
            node.chain = new_chain

    def validate_transactions(self, balances, new_transactions):
        assert self.check_balances(balances)

        validated = []
        for transaction in new_transactions:
            print('processing to:{} sender:{} amount:{}'.format(
                transaction.to, transaction.sender, transaction.amount))

            amount = float(transaction.amount)
            new_bal = balances[transaction.sender] - amount

            if new_bal < self.overdraft_limit:
                print('rejected - {} overdrawn with bal of {}'.format(
                    transaction.sender, new_bal))

            else:
                print('accepted')
                balances[transaction.sender] -= amount
                balances[transaction.to] += amount
                validated.append(transaction)

        assert self.check_balances(balances)

        return balances, validated

    def check_balances(self, balances):
        """ checks that all balances are over the limit """
        for node, balance in balances.items():
            assert balance >= self.overdraft_limit
        return True
```

Putting it all together - the code below simulates three blocks being added with random transactions

```python
if __name__ == '__main__':

    net = Network([Node('node'), Node('other')])
    transactions = net[0].chain[-1].transactions

    for _ in range(3):
        #  simulate a few transactions between nodes
        new_transactions = simulate_transactions(net[0], net[1])

        #  check the transactions are valid
        transactions = validate_transactions(
            transactions, new_transactions)

        new_proof = net.proof_of_work()
        #  randomly select a miner
        miner = net[np.random.randint(len(net))]

        #  miner adds the block to it's chain
        miner.add_next_block(new_transactions, proof=new_proof)

        #  update other nodes in the network
        net.consensus()
```

![]({{ "/assets/blockchain_python/fig1.png"}}) 

Next step for this work is to distribute the blockchain over multiple processes using Flask.

Thanks for reading!

## Resources and references

Brett Slatkin's excellent 'Effective Python' - I highly recommend this book. 

ecomusing - [blog post](http://ecomunsing.com/build-your-own-blockchain) - [GitHub](https://github.com/emunsing/tutorials/blob/master/BuildYourOwnBlockchain.ipynb)

Gerald Nash - [blog post one](https://medium.com/crypto-currently/lets-build-the-tiniest-blockchain-e70965a248b) - [blog post two](https://medium.com/crypto-currently/lets-make-the-tiniest-blockchain-bigger-ac360a328f4d)

Adil Moujahid - [blog post](http://adilmoujahid.com/posts/2018/03/intro-blockchain-bitcoin-python/) - [GitHub](https://github.com/adilmoujahid/blockchain-python-tutorial)
