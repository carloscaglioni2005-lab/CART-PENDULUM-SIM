# Simulatore Carrello-Pendolo

Applicazione Streamlit che simula un carrello che si muove su una guida orizzontale sotto l'effetto di una forza scelta dall'utente. Sotto il centro del carrello e collegato un pendolo tramite un'asta rigida.

L'app permette di:
- impostare parametri fisici del sistema
- definire le condizioni iniziali
- scegliere una legge di forza `F(t)`
- visualizzare i grafici temporali di posizione, velocita, angolo, energia e forza
- osservare la configurazione istantanea del sistema

## Avvio

```bash
cd /Users/carloscaglioni/vscode
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L'app locale e disponibile di default su `http://localhost:8501`.

## Pubblicazione su Internet

Il modo piu semplice per rendere pubblica questa app e usare Streamlit Community Cloud.

Struttura gia pronta per il deploy:
- entrypoint: `app.py`
- dipendenze: `requirements.txt`
- configurazione Streamlit: `.streamlit/config.toml`

Passi:
1. carica questa cartella in un repository GitHub pubblico o privato
2. vai su `https://share.streamlit.io/`
3. collega il tuo account GitHub
4. clicca `Create app`
5. seleziona repository, branch e file `app.py`
6. opzionale: scegli un sottodominio personalizzato
7. clicca `Deploy`

Note utili:
- Streamlit Community Cloud genera un URL pubblico su dominio `streamlit.app`
- i file sensibili come `.env` non devono essere caricati nel repository
- se in futuro aggiungi chiavi segrete, inseriscile nelle impostazioni `Secrets` del deploy e non nel repo

## Forza applicata

Nel campo `F(t)` puoi usare espressioni come:

```text
0
2 * Heaviside(t - 1)
1.8 * sin(1.6 * t)
3 * exp(-0.5 * t) * cos(4 * t)
```

Funzioni supportate: `sin`, `cos`, `tan`, `exp`, `sqrt`, `pi`, `Heaviside`, `Abs`.

## Modello

L'angolo `theta = 0` corrisponde al pendolo verticale verso il basso.

Equazioni del moto usate:

```text
(M + m) x_ddot + m l cos(theta) theta_ddot = F - C_m x_dot + m l sin(theta) theta_dot^2
m l cos(theta) x_ddot + m l^2 theta_ddot = tau - m g l sin(theta) - C_p theta_dot
```
