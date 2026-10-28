# Two-step proximal stationary-target contrast

2026-10-28. One CPU float64 enumeration; no model, optimizer, rollout, GPU or configuration change.

The deterministic binary tree has reward1 only on LR, reference probabilities .5 at each node, beta=.7, and behavior probabilities root-L=.2 / left-child-R=.15 / right-child-R=.7. Three independent logits make the Gibbs target representable; all paths have positive behavior probability. A bounded logit box [-5,5] contains the target and bounds the coverage/log-ratio quantities.

[Proximal KL §3 Eq3 and exact population conditions](https://arxiv.org/html/2610.18738v1) use current-token IS with the full future log-ratio cost. Their stationary-target guarantee requires coverage and local expressiveness; it is not an unbiased-gradient guarantee or an automatic guarantee for clipped/windowed/group-normalized practice.

[Anatomy §4 and Remark1](https://arxiv.org/html/2610.25725v1) center the complete trajectory regularized return. Deterministic pathwise constancy is the relevant mechanism; its finite-sample claims additionally require coverage and bounded log ratios.

Our independent enumeration gives pi*(LL,LR,RL,RR)=[0.13941685502743967, 0.581749434917681, 0.13941685502743967, 0.13941685502743967]; conditional probabilities=[0.7211662899451208, 0.8066786301976913, 0.5]. The exact regularized-objective gradient is [-9.020562075079397e-17, 1.3877787807814457e-17, 0.0]. Every trajectory has R−beta log(pi*/q)=0.408794760920568 up to floating-point error.

In coordinate order root-L / left-child-R / right-child-R, ProxKL full-future population field is [-9.376488288093401e-17, 1.0388238121721365e-17, 0.0]; using only the current-token cost gives [-0.10359479451340223, 1.0388238121721365e-17, 0.0]. Full-future credit also cancels with a zero history baseline. The trajectory-return group-centered field over all16 ordered pairs is [-7.853568977269846e-17, 4.88224737390556e-17, 1.669775429036235e-17].

For a distinct comparison, we map the whole two-token response to one focal action and call the actual production fixed-scale reducer. The current PPO/k3/TIS arithmetic uses frozen pi* old, q reference, two-sample reward-only advantages, mixed-group backward/KL, scale128 and complete action denominator2. At this anchor PPO clipping is inactive. Its ascent fields are:

| Sequence TIS | PG field | beta-k3 field | Total field |
|---|---|---|---|
| uncapped | [0.000633638098673541, 0.0004393148345312702, -0.0] | [-0.0002920145100553503, -0.0002560785314995568, -0.0] | [0.0003416235886181907, 0.00018323630303171342, -0.0] |
| cap2 | [8.240024423049517e-05, 5.7129850201931726e-05, -0.0] | [-5.534049189615949e-05, -5.266276179392677e-05, -0.0] | [2.7059752334335685e-05, 4.467088408004953e-06, -0.0] |

These nonzero fields describe different declared surrogates. They establish no current implementation bug, unbiasedness guarantee, new algorithm adoption, 31B conclusion or real task gain. The comparison uses synthetic probability tables, not a collector-admitted pair. Full inputs, every path/pair, normalization and source hashes are in [the JSON](proximal-stationarity-two-step-20261028.json).
