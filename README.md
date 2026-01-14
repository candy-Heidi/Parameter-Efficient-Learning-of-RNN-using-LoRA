# Parameter-Efficient-Learning-of-RNN-using-LoRA

This repository accompanies a research paper that explores  
**parameter-efficient fine-tuning of Recurrent Neural Networks (RNNs) using LoRA (Low-Rank Adaptation)**.

The code in this repository is provided as a reference implementation.  
The main purpose of this README is to summarize the **research motivation, methodology, and findings** of the paper.

---

## 📌 Research Motivation

As data continuously grows and evolves, models often need to be retrained to maintain performance.  
However, **fine-tuning all parameters of a neural network whenever new data arrives is computationally expensive**, especially in terms of training time and GPU memory.

Parameter-Efficient Fine-Tuning (PEFT) methods have been proposed to address this issue by updating only a small subset of model parameters.  
Among them, **LoRA (Low-Rank Adaptation)** has shown strong performance on Transformer-based models.

However, **research on applying LoRA to non-Transformer architectures, particularly RNNs, has been limited**.

---

## 🔍 Key Idea

This paper proposes a method to apply **LoRA to the hidden-state weight matrix of an RNN**.

Instead of re-training all RNN parameters when new data is introduced:

- The pre-trained RNN weights are **frozen**
- Only the **low-rank LoRA matrices** applied to the hidden-state transition are trained

This approach enables **efficient incremental learning** while preserving model performance.

---

## 🧠 Method Overview

The training process consists of two stages:

1. **RNN Pre-training Stage**
   - The RNN is trained normally on an initial dataset
   - All weight matrices are learned in this phase

2. **LoRA-based Incremental Learning Stage**
   - The hidden-state weight matrix is augmented with low-rank matrices
   - Original weights remain frozen
   - Only LoRA parameters are updated when new data is added

This design significantly reduces the number of trainable parameters during fine-tuning.

---

## 🧪 Experimental Results

Experiments were conducted on the **IWSLT 2017 English–German translation dataset**.

Compared to full-parameter fine-tuning, the proposed method achieved:

- **83.7% reduction in training time**
- **33.5% reduction in peak GPU memory usage**
- **Over 99.9% reduction in the number of trainable parameters**
- Comparable or slightly better performance in terms of perplexity

These results demonstrate that LoRA can be effectively applied to RNNs without sacrificing accuracy.

---

## 🎯 Contributions

- First systematic application of **LoRA to RNN hidden-state weights**
- Demonstration of **parameter-efficient incremental learning** for RNNs
- Significant reductions in training cost while maintaining performance
- Extension of PEFT techniques beyond Transformer architectures

---

## 📄 Paper Information

- **Title:** *Parameter-Efficient Learning of RNN using LoRA*  
- **Language:** Written in **Korean**
- **Paper Link:** The full paper can be accessed at the link below  

👉 **Paper:** [Link to the paper here]

---

## 🧠 One-line Summary

> **A parameter-efficient RNN fine-tuning method that applies LoRA to hidden-state transitions, enabling fast and memory-efficient incremental learning.**
