# KARL Deployment Runbook — first production deploy

Written for someone doing this for the first time. Every command says what it
does and what you should see. Work top to bottom; don't skip ahead.

**Target:** DigitalOcean droplet `159.203.91.194` (Ubuntu, nyc3) running the app
in Docker behind Caddy for HTTPS, talking to DO Managed Postgres 16 (nyc3).

Conventions below:
- `you@local$` — run on your own Windows machine (Git Bash).
- `root@droplet#` / `karl@droplet$` — run on the droplet, after you've SSH'd in.

A note on mistakes: almost everything here is reversible. The two commands that
are not are called out explicitly with **DESTRUCTIVE**. If something looks wrong,
stop and ask rather than improvising — a half-finished deploy is easy to fix, a
wiped database is not.

---

## Phase 0 — Get SSH access working

SSH is how you get a terminal on the droplet. It uses a **key pair**: a private
key (secret, stays on your machine) and a public key (safe to paste anywhere).
The server keeps the public half and uses it to recognise the private half.

A deploy key pair has already been generated on the dev machine:

```
~/.ssh/karl_deploy        <- private. Never send this to anyone, including me.
~/.ssh/karl_deploy.pub    <- public. Safe to paste.
```

Public key to install:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINbCiXOoDDB2FHJYJIWgpPwyjas/cAh9stqGFKBz3zYZ karl-deploy
```

### 0.1 Log in for the first time

When you created the droplet, DO either emailed you a root password or attached
an SSH key you already had. Easiest first entry is the DO web console:
**Droplets → karl → Access → Launch Droplet Console.** That drops you at a root
shell in the browser, no SSH needed.

### 0.2 Install the public key

At the root shell:

```bash
mkdir -p /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINbCiXOoDDB2FHJYJIWgpPwyjas/cAh9stqGFKBz3zYZ karl-deploy" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```

`authorized_keys` is the list of public keys allowed to log in as this user.
The `chmod` lines matter — SSH refuses to use these files if they're readable
by others, and fails with a confusing error.

### 0.3 Test it from your machine

```bash
you@local$ ssh -i ~/.ssh/karl_deploy root@159.203.91.194
```

First connection asks to confirm a fingerprint — type `yes`. You should land at
a `root@karl:~#` prompt. **Do not continue until this works.**

---

## Phase 1 — Basic droplet hygiene

### 1.1 Patch the system

```bash
root@droplet# apt update && apt upgrade -y
```

Takes a few minutes. If it asks about keeping local config files, accept the
default. If it says a reboot is required, do it (`reboot`) and SSH back in after
~30 seconds.

### 1.2 Create a non-root user

Running everything as root means any mistake is unlimited. Make a normal user:

```bash
root@droplet# adduser --disabled-password --gecos "" karl
root@droplet# usermod -aG sudo karl
root@droplet# mkdir -p /home/karl/.ssh
root@droplet# cp /root/.ssh/authorized_keys /home/karl/.ssh/
root@droplet# chown -R karl:karl /home/karl/.ssh
root@droplet# chmod 700 /home/karl/.ssh && chmod 600 /home/karl/.ssh/authorized_keys
```

Test in a **second terminal**, keeping the root session open as a lifeline:

```bash
you@local$ ssh -i ~/.ssh/karl_deploy karl@159.203.91.194
```

Once that works, use `karl@` from here on. Prefix admin commands with `sudo`.

### 1.3 Firewall

Allow only SSH and web traffic:

```bash
karl@droplet$ sudo ufw allow OpenSSH
karl@droplet$ sudo ufw allow 80/tcp
karl@droplet$ sudo ufw allow 443/tcp
karl@droplet$ sudo ufw --force enable
karl@droplet$ sudo ufw status
```

Note port 8000 is deliberately **not** opened — the app binds to loopback and is
reached only through Caddy. Confirm `OpenSSH` is in the list before enabling; a
firewall that blocks SSH locks you out of your own box (recoverable via the DO
web console, but avoid the detour).

---

## Phase 2 — Install Docker

```bash
karl@droplet$ curl -fsSL https://get.docker.com | sudo sh
karl@droplet$ sudo usermod -aG docker karl
```

The second line lets you run docker without `sudo`. **Log out and back in** for
it to take effect:

```bash
karl@droplet$ exit
you@local$ ssh -i ~/.ssh/karl_deploy karl@159.203.91.194
karl@droplet$ docker run --rm hello-world
```

Prints "Hello from Docker!" when working.

---

## Phase 3 — Get the code onto the droplet

The deploy branch is `test/m2-invite-flow-and-m1-corrections`, pushed to GitHub
on 2026-07-18. `main` is deliberately still at `4a94fd5` — the pilot runs from
the branch until it's proven in production, then it gets merged.

```bash
karl@droplet$ git clone https://github.com/tgraves719/Knowledge-Assistant-Rights-Labor.git karl
karl@droplet$ cd karl
karl@droplet$ git checkout test/m2-invite-flow-and-m1-corrections
karl@droplet$ git log --oneline -1
```

That last line should print `f707a0d docs: add first-time production deployment
runbook` (or newer). If it shows `4a94fd5`, the checkout didn't take and you're
on `main` — re-run the checkout.

Because the app is deployed from a branch, later updates are
`git pull origin test/m2-invite-flow-and-m1-corrections`, not a bare `git pull`.

---

## Phase 4 — Configuration

```bash
karl@droplet$ cp .env.example .env
karl@droplet$ nano .env
```

`nano` is a basic text editor: arrow keys to move, type to edit, `Ctrl+O` then
Enter to save, `Ctrl+X` to quit.

Fill in each required value. `.env.example` documents every one, but the two
worth repeating:

- `KARL_SECRET_ENCRYPTION_KEY` — a **passphrase seed**, not a Fernet key. Only
  the first 32 bytes are used and there's no key stretching, so use 32
  characters of real randomness. Generate one on the droplet:

  ```bash
  karl@droplet$ python3 -c "import secrets; print(secrets.token_urlsafe(32)[:32])"
  ```

- `KARL_POSTGRES_URL` — note the `+psycopg` suffix and `sslmode=require`, and
  point at the **`karl`** database, not `defaultdb`:

  ```
  postgresql+psycopg://doadmin:<PASSWORD>@karl-db-pgsql-nyc3-06363-do-user-40355769-0.e.db.ondigitalocean.com:25060/karl?sslmode=require
  ```

Lock the file down — it holds every secret you have:

```bash
karl@droplet$ chmod 600 .env
```

`.env` is gitignored, so it will never be committed.

---

## Phase 5 — Build and start

```bash
karl@droplet$ docker compose -f docker-compose.prod.yml build
```

First build takes several minutes (it installs Python dependencies). Then:

```bash
karl@droplet$ docker compose -f docker-compose.prod.yml up -d
```

`-d` means detached — it runs in the background. The container runs
`alembic upgrade head` on start, which creates the schema in the `karl`
database, then serves the app.

Watch it come up:

```bash
karl@droplet$ docker compose -f docker-compose.prod.yml logs -f
```

`Ctrl+C` stops watching (it does **not** stop the app). Look for uvicorn
reporting it's running on `0.0.0.0:8000`.

If compose refuses to start and names a variable, that's the intended guard —
that variable is missing from `.env`.

### 5.1 Verify locally on the box

```bash
karl@droplet$ curl -i http://127.0.0.1:8000/api/health
karl@droplet$ docker compose -f docker-compose.prod.yml ps
```

Expect `200 OK` from the first, and `healthy` in the second once the 40-second
start period has passed. **Don't move on until health is green.**

---

## Phase 6 — Domain and HTTPS

### 6.1 Point DNS at the droplet

At your domain registrar for `karlstewardship.com`, create:

| Type | Name | Value |
|---|---|---|
| A | `@` | `159.203.91.194` |
| A | `www` | `159.203.91.194` |

DNS propagation is usually minutes. Check from your machine:

```bash
you@local$ nslookup karlstewardship.com
```

Wait until it returns the droplet IP before continuing — Caddy needs working DNS
to obtain a certificate, and repeated failures can hit rate limits.

### 6.2 Install Caddy

Caddy is a web server that fetches and renews Let's Encrypt certificates
automatically. It listens on 80/443 and forwards to the app on 8000.

```bash
karl@droplet$ sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
karl@droplet$ curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
karl@droplet$ curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
karl@droplet$ sudo apt update && sudo apt install -y caddy
```

### 6.3 Configure it

```bash
karl@droplet$ sudo nano /etc/caddy/Caddyfile
```

Replace the contents with:

```
karlstewardship.com, www.karlstewardship.com {
    encode gzip
    reverse_proxy 127.0.0.1:8000

    header {
        # HSTS: tells browsers to only ever use HTTPS for this domain.
        # Add this only AFTER confirming HTTPS works — it is sticky, and
        # browsers will refuse plain HTTP for max-age afterwards. Deploy the
        # block without this line first, verify, then add it and reload.
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        # ">" REPLACES rather than appends. The app already sets these three
        # itself; without ">" every response carries two copies, and a
        # duplicated X-Frame-Options can be treated as invalid.
        >X-Content-Type-Options "nosniff"
        >X-Frame-Options "DENY"
        >Referrer-Policy "strict-origin-when-cross-origin"
    }
}
```

Then:

```bash
karl@droplet$ sudo systemctl reload caddy
karl@droplet$ sudo systemctl status caddy
```

### 6.4 Confirm

Visit `https://karlstewardship.com` — the org landing page, with a padlock.

```bash
you@local$ curl -I https://karlstewardship.com/api/health
```

Also confirm `KARL_ALLOWED_ORIGINS=https://karlstewardship.com` is set in `.env`.
If you changed it, restart:

```bash
karl@droplet$ docker compose -f docker-compose.prod.yml up -d
```

The session cookie is `secure`, meaning it is only sent over HTTPS — the join
flow will not work over plain HTTP. This phase is a prerequisite for testing it.

---

## Phase 7 — Post-deploy hardening

Do these only after everything above works, so you're never debugging
connectivity and deployment simultaneously.

1. **Rotate the database password.** DO console → Databases → karl-db → Users →
   reset `doadmin`. Update `.env`, then `docker compose ... up -d`.
2. **Restrict trusted sources.** DO console → Databases → Settings → Trusted
   Sources → add the droplet, remove everything else. This closes the database
   to the public internet.
3. **Switch to the private hostname.** DO console → Databases → Connection
   Details → **VPC network**. Swap that host into `KARL_POSTGRES_URL` so traffic
   stays inside DO's network. Restart after.
4. **Confirm backups** are on, and do a **test restore** to a scratch database.
   The pilot plan requires a *tested* restore, not merely enabled backups — an
   untested backup is a hope, not a recovery plan.

---

## Everyday operations

```bash
# Look at logs
docker compose -f docker-compose.prod.yml logs -f

# Restart after changing .env
docker compose -f docker-compose.prod.yml up -d

# Deploy new code (note the explicit branch — the app runs from a branch)
git pull origin test/m2-invite-flow-and-m1-corrections
docker compose -f docker-compose.prod.yml up -d --build

# Stop the app (data is safe; it lives in the managed database)
docker compose -f docker-compose.prod.yml down
```

**DESTRUCTIVE — know before you type:**

- `docker compose ... down -v` also deletes named volumes, including uploaded
  documents. The managed database is untouched, but uploads are gone.
- Anything with `DROP DATABASE` or `DROP TABLE` against the `karl` database.
  There is no undo except a restore, which is why step 7.4 exists.

---

## When something breaks

| Symptom | Likely cause |
|---|---|
| Compose exits naming a variable | That variable is missing from `.env` |
| `password authentication failed` | Password wrong in `KARL_POSTGRES_URL`, or special characters need URL-encoding |
| App starts, `/api/health` hangs | Check `KARL_POSTGRES_URL` host/port; trusted sources may exclude the droplet |
| Caddy won't get a certificate | DNS not pointing at the droplet yet, or ports 80/443 blocked |
| `permission denied` on docker | You skipped the log-out/log-in after `usermod -aG docker` |

Logs first, always:

```bash
docker compose -f docker-compose.prod.yml logs --tail=100
sudo journalctl -u caddy --no-pager -n 50
```
