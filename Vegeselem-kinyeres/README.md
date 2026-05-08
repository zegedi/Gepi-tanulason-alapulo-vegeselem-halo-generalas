# Végeselem-kinyerés

A projekt ezen része az [SRL-Assisted AFM](https://github.com/CMU-CBML/SRL-AssistedAFM) algoritmussal létrehozott kimeneteket, illetve a generáláshoz használt fájlokat tartalmazza. 

## 📂 Adatok

Az `AFM_data.zip` fájl az alábbi struktúrát tartalmazza:

* **`input/`**: A generáláshoz szükséges bemeneti szöveges fájlok helye.
* **`output/`**: A generált kimeneti állományok gyűjtőhelye.
* **`config/`**: A bemeneti fájlok generálásához használt konfigurációk.
* **`gmsh/`**: A geometriákat és a hozzájuk tartozó referencia hálókat tartalmazza.

---

## 🛠 Telepítés

A projekt futtatásához klónozzuk le az [SRL-Assisted AFM](https://github.com/CMU-CBML/SRL-AssistedAFM) könyvtárat:

```bash
git clone https://github.com/CMU-CBML/SRL-AssistedAFM

```

A példák generálása az itt található `inference.py` program futtatásával végezhető el.

A futtatáshoz szükséges programkönyvtárak az alábbi utasítással telepíthetőek:

```bash
pip install -r requirements.txt

```

---

## 📄 Bemeneti formátum leírása

Az `inference.py` program egy meghatározott szerkezetű szöveges fájlt vár bemenetként, amelynek három fő rész van.

### 1. Alap információk

Ez a rész egyetlen sort tartalmaz, amely az alábbi adatokat definiálja:

```text
total_front_length initial_number_of_nodes

```

* **`total_front_length`**: A kiindulási perem élhosszainak az összege (lebegőpontos).
* **`initial_number_of_nodes`**: A kezdeti peremvonalon található csúcsok száma (páros, pozitív egész).

### 2. Perempontok definíciója

Ez a szakasz pontosan `initial_number_of_nodes` számú sort tartalmaz (minden csúcshoz egyet), az alábbi felépítéssel:

```text
x_coordinate y_coordinate inner_angle reference_priority left_edge_length total_reference_length

```

**Paraméterek részletezése:**

1. **`x_coordinate`**: A csúcs X koordinátája (lebegőpontos).
2. **`y_coordinate`**: A csúcs Y koordinátája (lebegőpontos).
3. **`inner_angle`**: A csúcs belső szöge a bal és jobb oldali szomszédos élek függvényében (nemnegatív lebegőpontos).
4. **`left_edge_length`**: Az aktuális csúcs és a bal oldali szomszédja közötti euklideszi távolság (pozitív lebegőpontos).
5. **`total_reference_length`**: Az aktuális ponthoz képesti három-három bal- és jobboldali szomszédos él hosszának az összege (pozitív lebegőpontos).

### 3. Laplace simítás fixpontjai

Ez a szekció a hálósimításhoz használt rögzített pontokat határozza meg. A szakasz legfeljebb `initial_number_of_nodes` darab sort tartalmazhat. Minden sor a fixpont koordinátáit adja meg:

```text
x_coordinate y_coordinate

```

* **`x_coordinate`**: A fixpont X koordinátája (lebegőpontos).
* **`y_coordinate`**: A fixpont Y koordinátája (lebegőpontos).

---

## 🚀 Használati útmutató

### 1. Bemenet létrehozása

A bemeneti adatokat a geometria típusa alapján az alábbi szkriptekkel lehet létrehozni:

* **Többszörösen összefüggő geometria:**
`input_gmsh_boundary_multi_connected.py`
* **Egyszeresen összefüggő geometria:**
`input_gmsh_boundary_single_connected.py`


### 2. Háló generálása

A telepítési utasításokat követően az [SRL-Assisted AFM](https://github.com/CMU-CBML/SRL-AssistedAFM) könyvtárhoz tartozó `inference.py` fájl segítségével lehet hálót generálni az alábbi módon:

```bash
python inference.py
```



### 3. Kimenet konvertálása

Az [SRL-Assisted AFM](https://github.com/CMU-CBML/SRL-AssistedAFM) könyvtárhoz tartozó `inference.py` futtatásával  kapott `.inp` formátumú kimenetet `.msh` formátumba lehet alakítani a `convert_to_msh.py` segítségével.
Ehhez szükség van a hálóhoz tartozó geometriára `.step` formátumban.
