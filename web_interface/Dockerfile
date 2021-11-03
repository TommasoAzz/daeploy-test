# Angular build and deploy on nginx
## STAGE 1: Build ###
FROM node:16.13.0-alpine AS build
WORKDIR /usr/src/app
RUN npm install -g @angular/cli
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

## STAGE 2: Run ###
FROM nginx:1.21.3-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /usr/src/app/dist/iris-classifier /usr/share/nginx/html

# Angular fast deploy and hot-reload
# FROM node:12-alpine

# RUN apk add --update-cache \
#     git  \
#     && rm -rf /var/cache/apk/*

# COPY ./docker-entrypoint.develop.sh /docker-entrypoint.sh

# WORKDIR /app
# COPY . .
# RUN chmod +x /docker-entrypoint.sh

# EXPOSE 4200

# ENV SUBFOLDER=""
# USER node

# ENTRYPOINT /docker-entrypoint.sh