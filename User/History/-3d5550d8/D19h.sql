DROP TABLE IF EXISTS companies;
DROP TABLE IF EXISTS "user";
DROP TABLE IF EXISTS medical_kit;
DROP TABLE IF EXISTS product;
DROP TABLE IF EXISTS product_type;
DROP TABLE IF EXISTS temp_auto_removed;

CREATE TABLE companies (
    index_id SERIAL PRIMARY KEY,
    company_name TEXT NOT NULL,
    users_sha256 TEXT DEFAULT NULL
);

CREATE TABLE "user" (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    badge_id TEXT,
    clearance INTEGER NOT NULL,
    company_id INTEGER NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies (index_id)
);

CREATE TABLE medical_kit (
    mac TEXT PRIMARY KEY,
    company_id INTEGER NOT NULL,
    alias TEXT,
    last_update TEXT DEFAULT CURRENT_TIMESTAMP,
    api_token_hash TEXT NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies (index_id)
);

CREATE TABLE product_type (
    type_name TEXT PRIMARY KEY,
    number INTEGER NOT NULL
);

CREATE TABLE product (
    product_id TEXT PRIMARY KEY,
    mac TEXT NOT NULL,
    product_type TEXT NOT NULL,
    expiration_date TEXT NOT NULL,
    FOREIGN KEY (mac) REFERENCES medical_kit (mac),
    FOREIGN KEY (product_type) REFERENCES product_type (type_name)
);

CREATE TABLE kit__ongoing_registration (
    mac TEXT PRIMARY KEY,
    company_id INTEGER NOT NULL,
    api_token TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mac) REFERENCES medical_kit (mac),
    FOREIGN KEY (company_id) REFERENCES companies (index_id)
);

INSERT INTO product_type (type_name, number) VALUES ('guanti_sterili_monouso', 5);
INSERT INTO product_type (type_name, number) VALUES ('visiera_paraschizzi', 1);
INSERT INTO product_type (type_name, number) VALUES ('soluzione_cutanea_iodopovidone', 1);
INSERT INTO product_type (type_name, number) VALUES ('soluzione_fisiologica', 3);
INSERT INTO product_type (type_name, number) VALUES ('compresse_garza_10x10', 10);
INSERT INTO product_type (type_name, number) VALUES ('compresse_garza_18x40', 2);
INSERT INTO product_type (type_name, number) VALUES ('teli_sterili_monouso', 2);
INSERT INTO product_type (type_name, number) VALUES ('pinzette_medicazione', 2);
INSERT INTO product_type (type_name, number) VALUES ('rete_elastica', 1);
INSERT INTO product_type (type_name, number) VALUES ('cotone_idrofilo', 1);
INSERT INTO product_type (type_name, number) VALUES ('cerotti_pronti_uso', 2);
INSERT INTO product_type (type_name, number) VALUES ('rotoli_cerotto', 2);
INSERT INTO product_type (type_name, number) VALUES ('forbici', 1);
INSERT INTO product_type (type_name, number) VALUES ('lacci_emostatici', 3);
INSERT INTO product_type (type_name, number) VALUES ('ghiaccio_pronto_uso', 2);
INSERT INTO product_type (type_name, number) VALUES ('sacchetti_rifiuti_sanitari', 2);
INSERT INTO product_type (type_name, number) VALUES ('termometro', 1);
INSERT INTO product_type (type_name, number) VALUES ('misurazione_pressione_arteriosa', 1);


CREATE TABLE temp_auto_removed (
    mac TEXT NOT NULL,
    item_id TEXT PRIMARY KEY,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mac) REFERENCES medical_kit (mac),
    FOREIGN KEY (item_id) REFERENCES product (product_id)
);