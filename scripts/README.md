## Term-description similarities

### Average similarity
Given a matrix *__A__*, where each row is a word in a term and each column is a word in a description and each element is the cosine similarity between each term word and description word, respectively.
We compute the average similarity between each term and description as the sum of the maximum similarities across each column and each row:

$$AvgSim = \frac{\sum max(A_i) + \sum max(A_j)}{m+n}$$

------------------------
### Weighted similarities

#### Individual
We can also compute a weighted average similarity score where each of the maximum row and column values are weighted by their respective tfidf where:

$$f(x) = tfidf(w)$; $w = \text{a word in a term name or sample description}$$

$$AvgSim_{weighted} = \frac{\sum f(t_i)max(A_i) + \sum f(d_j)max(A_j)}{\sum f(t_i) + \sum f(d_j)}$$

Here, $t_i$ is a word in a term name representing the row $\mathbf{A}_i$ and $d_j$ is a word in a sample description representing the column $\mathbf{A}_j$. This method is termed individual weighting, since we are only weighting a maximum score by the tfidf of each row and column, respectively.

#### Product
Further, we can compute another weighted average termed, product weighting, where we also multiply the tfidf of the word that produced the maximum similarity for a given row or column. 

For example, if the maximum similarity for $\mathbf{A_{i=1}}$ was found at column 3 of $\mathbf{A}$, then we would multiply that maximum similarity by both the tfidf for the term word representing row 1 and the tfidf of the word representing column 3. 

------------------------
### Transformations

#### Single_z
We also apply two transformation methods: single_z and double_z. Where for single_Z we compute a zscore for each term across all sample  where for a matrix $\mathbf{A}$ with terms as rows and samples as columns:

$$Z_{ij}^{S} = \frac{A_{ij} - \mu_i}{\sigma_i};$ For all $i$ where $\max(A_i) \neq min(A_i)$$

Here, $\mathbf{Z}^{S}$ is the resulting matrix of Z-scores for each similarity distribution of terms.


#### Double_z
For double_z, we combine the Z-score distributions of both the terms and the samples as:

$$\mathbf{Z}_{ij}^{terms} = \frac{\mathbf{A}_{ij} - \mu_i}{\sigma_i};$ For all $i$ where $\max(\mathbf{A}_i) \neq min(\mathbf{A}_i)$$

$$\mathbf{Z}_{ij}^{samples} = \frac{\mathbf{A}_{ij} - \mu_j}{\sigma_j};$ For all $j$ where $\max(\mathbf{A}_j) \neq min(\mathbf{A}_j)$$

Then,
$$\mathbf{Z}^{D}=\frac{\mathbf{Z}^{terms} + \mathbf{Z}^{samples}}{\sqrt{2}}$$

Here, $\mathbf{Z}^{D}$ is the resulting matrix of added Z-scores for each similarity distribution of terms and samples.
