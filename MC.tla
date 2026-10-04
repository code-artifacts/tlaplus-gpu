---- MODULE MC ----
EXTENDS PaxosCommit, TLC

\* MV CONSTANT declarations@modelParameterConstants
CONSTANTS
a1, a2, a3
----

\* MV CONSTANT declarations@modelParameterConstants
CONSTANTS
r1, r2
----

\* MV CONSTANT definitions Acceptor
const_17706902386892000 == 
{a1, a2, a3}
----

\* MV CONSTANT definitions RM
const_17706902386903000 == 
{r1, r2}
----

\* SYMMETRY definition
symm_17706902386904000 == 
Permutations(const_17706902386892000) \union Permutations(const_17706902386903000)
----

\* CONSTANT definitions @modelParameterConstants:0Ballot
const_17706902386905000 == 
{0,1}
----

\* CONSTANT definitions @modelParameterConstants:2Majority
const_17706902386906000 == 
{{a1,a2},{a2,a3},{a1,a3}}
----

=============================================================================
\* Modification History
\* Created Tue Feb 10 10:23:58 CST 2026 by lsr
