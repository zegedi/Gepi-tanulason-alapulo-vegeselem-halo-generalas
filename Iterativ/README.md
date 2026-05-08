# Iteratív hálózás

Ez a projekt a Kohonen- és Nechaeva-féle iteratív algoritmusok segítségével végez hálógenerálást.

## 📂 Projekt felépítése és adatok

A projekt tartalmaz egy tömörített **zip** fájlt, 
amely magában foglal néhány példa bemenetet és eredményt.

### Bemenetek (`test_json/`)

A bemeneti JSON fájlok két fő kategóriára oszlanak:

* **`1991/`**: Az eredeti **Kohonen-módszerhez** tartozó bemeneti állományok.
* **`2006/`**: A **Nechaeva-féle módszerhez** tartozó bemeneti állományok.

### Kimenetek és Geometriák (`test_geo/`)

Ebben a mappában találhatók a generált eredmények:

* **`1991/`** és **`2006/`**: A megfelelő módszerekkel generált kimeneti fájlok.
* **`gmsh/`**: Referencia hálók, amelyek a **Frontal-Delaunay** algoritmussal készültek.
* **`./`**: A mappa közvetlenül tartalmazza a vizsgálathoz használt alapgeometriákat.

---

## 🛠 Telepítés

A projekt futtatásához legalább Python 3.12 szükséges. 
A függőségek telepítése az alábbi `pip` utasítással lehetséges:

```bash
pip install -r requirements.txt

```

---

## 🚀 Használat

A hálógeneráló főprogram a parancssorból indítható el. 
Paraméterként egy konfigurációs JSON fájl elérési útját kell magadni.

**Szintaxis:**

```bash
python main.py path_to_input_json

```

**Példa:**

```bash
python main.py test_json/2006/arc_geometry.json

```
