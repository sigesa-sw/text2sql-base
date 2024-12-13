CREATE TABLE IF NOT EXISTS costes (
    HOSPITAL VARCHAR(100),
    AÑO INTEGER,
    MES INTEGER,
    SERVICIO VARCHAR(100),
    FINANCIADOR VARCHAR(100),
    ALTAS INTEGER,
    COSTE_UNIDAD FLOAT,
    COSTES FLOAT,
    TARIFA_UNIDAD FLOAT,
    INGRESOS FLOAT,
    RENTABILIDAD FLOAT
);

COPY costes FROM '/docker-entrypoint-initdb.d/VisualCost_Tablas_2024.csv' DELIMITER ';' CSV HEADER;