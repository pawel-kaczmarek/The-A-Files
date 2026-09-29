# syntax=docker/dockerfile:1
#
# The A-Files web client (Next.js standalone server).
#
#   docker build -t the-a-files-web --build-arg NEXT_PUBLIC_TAF_API_URL=http://localhost:8000 web
#
# The browser calls the API directly, so NEXT_PUBLIC_TAF_API_URL is the API
# address as seen from the browser. Next.js inlines it at build time.

ARG NODE_VERSION=22

FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci


FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app

ARG NEXT_PUBLIC_TAF_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_TAF_API_URL=$NEXT_PUBLIC_TAF_API_URL \
    NEXT_TELEMETRY_DISABLED=1 \
    TAF_NEXT_OUTPUT=standalone

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build


FROM node:${NODE_VERSION}-alpine

LABEL org.opencontainers.image.source="https://github.com/pawel-kaczmarek/The-A-Files" \
      org.opencontainers.image.description="The A-Files research platform web client" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static

USER node
EXPOSE 3000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:3000/dashboard"]

CMD ["node", "server.js"]
