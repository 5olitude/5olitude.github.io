---
layout: post
title:  "Welcome to Blockchain Introduction!"
date:   2024-11-27 17:39:17 +0000
categories: blockchain technology
slug: blockchain-introduction
permalink: /blockchain/technology/:year/:month/:day/:slug/
---

## Introduction to Blockchain Technology

Blockchain technology is revolutionizing industries worldwide by providing a decentralized, secure, and transparent system for transactions. In this post, we will explore some of the key components of blockchain technology, including:

### 1. Blockchain Address Generation

A blockchain address is a unique identifier used for sending and receiving cryptocurrency. It's typically generated using a public key from the public-key cryptography system. This ensures that only the holder of the corresponding private key can access and control the funds.
#### How Blockchain Addresses are Generated (Example using golang) 
  Before jumping into it, we must understand how a wallet is structured programmatically.

{% highlight go %}
type Wallet struct {
    privateKey        *ecdsa.PrivateKey
    publicKey         *ecdsa.PublicKey
    blockchainAddress string
      }
  
{% endhighlight %}
 So to define and generate privateKey and PublicKey into the wallet struct we must use the [Golang CRYPTO  ECDSA ](https://pkg.go.dev/crypto/ecdsa) package , and to know how bitcoin wallet address are generated please visit [Bitcoin Foundation](https://en.bitcoin.it/wiki/Technical_background_of_version_1_Bitcoin_addresses)

 For Easy Understanding im attaching the steps here along with the code :

#### 1* *Creating ECDSA private key (32 bytes) public key (64 bytes)*
    privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)

  *P256 refers to the NIST P-256 curve, which is a widely used elliptic curve for ECDSA and other cryptographic protocols.*

  *Rand Reader It is used to generate secure random numbers, which are essential for generating cryptographic keys*

  here we are generating the private key using ecdsa.GenerateKey and the genarated private key conatins the Public Key (32 X bytes, 32 Y bytes ) 64 bytes in length  and it is in struct which looks like :

  ```go

   type PrivateKey struct {
    PublicKey
    D *big.Int
}


  ```
#### 2* *Perform SHA-256 hashing on the public key (32 bytes)*
 ```go

  h2 := sha256.New()
  h2.Write(w.publicKey.X.Bytes())
  h2.Write(w.publicKey.Y.Bytes())
  digest2 := h2.Sum(nil)

```
#### 3* *Perform RIPEMD-160 hashing on the result of SHA-256 (20 bytes)*
 ```go
 
h3 := ripemd160.New()
h3.Write(digest2)
digest3 := h3.Sum(nil)

 ```
#### 4* *Add version byte in front of RIPEMD-160 hash (0x00 for Main Network)*
```go
vd4 := make([]byte, 21)
vd4[0] = 0x00
copy(vd4[1:], digest3[:])

```
####  5* *Perform SHA-256 hash on the extended RIPEMD-160 result*
```go
	h5 := sha256.New()
	h5.Write(vd4)
	digest5 := h5.Sum(nil)

  ```
#### 6*  *Perform SHA-256 hash on the result of the previous SHA 256 hash*
```go
	h6 := sha256.New()
	h6.Write(digest5)
	digest6 := h6.Sum(nil)
  ```
#### 7*  *Take the first 4 bytes of the second SHA-256 hash for checksum*
 ```go
	chsum := digest6[:4]
  ```
#### 8*  *Add the 4 checksum bytes from 7th step at the end of extended RIPEMD-160 hash from step 4 (25 bytes)*
```go
	dc8 := make([]byte, 25)
	copy(dc8[:21], vd4[:])
	copy(dc8[21:], chsum[:])
```

#### 9*  *Convert the result from a byte string into base58*
```go
	address := base58.Encode(dc8)
	w.blockchainAddress = address
```
### 2. Transaction Verification and Signature Verification

The next steps which are  important is transaction verification , signature generation and signature verification 

#### 1* *Transaction Fields*
Internally in a node  when a user attempts to sends a bitcoin to another user the values which are necessary are :

  **Senders Private Key**

  **Senders Public  Key**

  **Senders Blockchain Address**

  **Recipients Blockchain Address**

  **Value to be sended**

in the end transaction struct looks like this :
   ```go
     type Transaction struct {
      senderPrivateKey           *ecdsa.PrivateKey
      senderPublicKey            *ecdsa.PublicKey
      senderBlockchainAddress    string
      recipientBlockchainAddress string
      value                      float32
      }

   ```
#### 2* *Signature Generation using transaction above*

```go

func (t *Transaction) GenerateSignature() *utils.Signature {
	m, _ := json.Marshal(t)
	h := sha256.Sum256([]byte(m))
	r, s, _ := ecdsa.Sign(rand.Reader, t.senderPrivateKey, h[:])
	return &utils.Signature{
		r,
		s,
	}

}

```
In the above code snippet we are converting the transaction into byte using json marshal method and taking the sha256 sum of the same. Using ecdsa.Sign method we are generating  signature, for this we need the **sha256 sum of transaction and SenderPrivate** key .

#### 3* *Signature Verification*

```go
func (bc *Blockchain) VerifyTransactionSignature(senderPublicKey *ecdsa.PublicKey, s *utils.Signature, t *Transaction) bool {
	m, _ := json.Marshal(t)
	h := sha256.Sum256([]byte(m))
	return ecdsa.Verify(senderPublicKey, h[:], s.R, s.S)
}

```
 The code above just verify the signature to ensure security , we are just creating **sha256 sum of transaction** and verify it along with the **senders public key** and **signature generated >  s.R & s.S**


#### 4*  *Adding Transaction*

   If Siganture is verified its now added to the blockchain transaction pool which is basically a list of transactions and now its the duty of the miner to create a block and inscribe the transactions from pool



### 3. Proof of Work: Ensuring Security

Proof of work is the consensus mechanism that underpins many blockchains, including Bitcoin. It requires miners to solve a computationally difficult puzzle before adding a new block to the chain. This ensures that only valid blocks are added, and the blockchain remains secure from malicious attacks.

**Proof of Work** is one of the energy consuming process and in the begining **Ethereum** worked on the concept of **Proof of work** later it shifted to **Proof of Stake** method 

#### 1 . *Nonce : Complexity and competition between miners*

```go 

func (bc *Blockchain) ProofOfWork() int {
	transactions := bc.CopyTransactionPool()
	previousHash := bc.LastBlock().Hash()
	nonce := 0
	for !bc.ValidProof(nonce, previousHash, transactions, MINING_DIFFICULTY) {
		nonce += 1
	}
	return nonce
}

func (bc *Blockchain) ValidProof(nonce int, previousHash [32]byte, transactions []*Transaction, difficulty int) bool {
	zeros := strings.Repeat("0", difficulty)
	guessBlock := Block{0, nonce, previousHash, transactions}
	guessHashStr := fmt.Sprintf("%x", guessBlock.Hash())
	return guessHashStr[:difficulty] == zeros
}


```
Going deep into code ,Checking  ProofOfWork method we have copied the transactions and caluclated the previous blocks hash inorder to create the new block but nonce is necessary to create the new block and we have given the difficult as "000" so we have to iterate through Valid proof method until we found a valid hash which begins with 000 and increment the nonce until the valid hash is found 

**if a valid nonce is found we create a new block by passing previous hash and nonce**

 ```go
  bc.CreateBlock(nonce, previousHash)

  func (bc *Blockchain) CreateBlock(nonce int, previousHash [32]byte) *Block {
	b := NewBlock(nonce, previousHash, bc.transactionPool)
	bc.chain = append(bc.chain, b)
	bc.transactionPool = []*Transaction{}
	return b
}
 ```
***The transactions in the transaction pools are emptied as everything is verified***

### Conclusion

Blockchain technology combines cryptography, decentralization, and consensus mechanisms to create a system that is secure, transparent, and trustworthy. Understanding the basics of address generation, mining, proof of work, and transaction verification is essential to grasp how this technology works.

---

Feel free to dig deeper into any of these topics as you explore the revolutionary world of blockchain technology!