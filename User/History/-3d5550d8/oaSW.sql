CREATE TABLE expirations (
    mac TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    PRIMARY KEY (mac, item_id)
);

CREATE TABLE companies (
    index_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    users_sha256 TEXT DEFAULT NULL
);

CREATE TABLE user (
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
    guanti_sterili_monouso TEXT,
    visiera_paraschizzi TEXT,
    soluzione_cutanea_iodopovidone TEXT,
    soluzione_fisiologica TEXT,
    compresse_garza_10x10 TEXT,
    compresse_garza_18x40 TEXT,
    teli_sterili_monouso TEXT,
    pinzette_medicazione TEXT,
    rete_elastica TEXT,
    cotone_idrofilo TEXT,
    cerotti_pronti_uso TEXT,
    rotoli_cerotto TEXT,
    forbici TEXT,
    lacci_emostatici TEXT,
    ghiaccio_pronto_uso TEXT,
    sacchetti_rifiuti_sanitari TEXT,
    termometro TEXT,
    misurazione_pressione_arteriosa TEXT,
    last_update TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies (index_id)
);

CREATE TABLE Product {
    product_id TEXT PRIMARY KEY,
    mac TEXT,
    product_type TEXT,
    expiration_date TEXT,
}


CREATE TABLE Product_type {
    type_name TEXT PRIMARY KEY,
    number INTEGER NOT NULL,
}

INSERT INTO Product_type (type_name, number) VALUES ("guanti_sterili_monouso", 5);
INSERT INTO Product_type (type_name, number) VALUES ("visiera_paraschizzi", 1);
INSERT INTO Product_type (type_name, number) VALUES ("soluzione_cutanea_iodopovidone", 1);
INSERT INTO Product_type (type_name, number) VALUES ("soluzione_fisiologica", 3);
INSERT INTO Product_type (type_name, number) VALUES ("compresse_garza_10x10", 10);
INSERT INTO Product_type (type_name, number) VALUES ("compresse_garza_18x40", 2);
INSERT INTO Product_type (type_name, number) VALUES ("teli_sterili_monouso", 2);
INSERT INTO Product_type (type_name, number) VALUES ("pinzette_medicazione", 2);
INSERT INTO Product_type (type_name, number) VALUES ("rete_elastica", 1);
INSERT INTO Product_type (type_name, number) VALUES ("cotone_idrofilo", 1);
INSERT INTO Product_type (type_name, number) VALUES ("cerotti_pronti_uso", 2);
INSERT INTO Product_type (type_name, number) VALUES ("rotoli_cerotto", 2);
INSERT INTO Product_type (type_name, number) VALUES ("forbici", 1);
INSERT INTO Product_type (type_name, number) VALUES ("lacci_emostatici", 3);
INSERT INTO Product_type (type_name, number) VALUES ("ghiaccio_pronto_uso", 2);
INSERT INTO Product_type (type_name, number) VALUES ("sacchetti_rifiuti_sanitari", 2);
INSERT INTO Product_type (type_name, number) VALUES ("termometro", 1);
INSERT INTO Product_type (type_name, number) VALUES ("misurazione_pressione_arteriosa", 1);


CREATE TABLE temp_auto_removed (
    mac TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (mac, item_id)
);