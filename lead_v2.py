resultado_prime = pd.DataFrame({
    "dt_movimento": exog_futuro["dt_movimento"].reset_index(drop=True),
    "previsto": np.asarray(previsto_prime),          # <- essa linha é a correção
    "limite_inferior_95": np.asarray(ic_prime)[:, 0],
    "limite_superior_95": np.asarray(ic_prime)[:, 1],
})
resultado_prime_setembro = resultado_prime[resultado_prime["dt_movimento"] >= "2026-09-01"].reset_index(drop=True)
resultado_prime_setembro



resultado_varejo = pd.DataFrame({
    "dt_movimento": exog_futuro["dt_movimento"].reset_index(drop=True),
    "previsto": np.asarray(previsto_varejo),         # <- correção
    "limite_inferior_95": np.asarray(ic_varejo)[:, 0],
    "limite_superior_95": np.asarray(ic_varejo)[:, 1],
})
resultado_varejo_setembro = resultado_varejo[resultado_varejo["dt_movimento"] >= "2026-09-01"].reset_index(drop=True)
resultado_varejo_setembro
