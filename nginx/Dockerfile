FROM nginx:1.27-alpine
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/certs \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
       -keyout /etc/nginx/certs/privkey.pem \
       -out /etc/nginx/certs/fullchain.pem \
       -subj "/C=CL/ST=RM/L=Santiago/O=PagoSeguro/CN=localhost"
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 443

