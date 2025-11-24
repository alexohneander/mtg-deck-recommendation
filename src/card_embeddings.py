"""
MTG Card Embedding Generator
Erstellt Vektorrepräsentationen von Magic The Gathering Karten für ML-Modelle
"""

import sqlite3
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re


class MTGCardEmbedder:
    """Klasse zum Erstellen von Feature-Vektoren für MTG Karten"""
    
    def __init__(self, db_path: str = 'data/AllPrintings.sqlite'):
        self.db_path = db_path
        self.feature_names = []
        self.color_map = {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4, 'C': 5}
        self.type_vocab = set()
        self.subtype_vocab = set()
        self.keyword_vocab = set()
        self.max_cmc = 20  # Maximum converted mana cost
        
    def connect_db(self) -> sqlite3.Connection:
        """Verbindung zur Datenbank herstellen"""
        return sqlite3.connect(self.db_path)
    
    def parse_mana_cost(self, mana_cost: Optional[str]) -> Dict[str, int]:
        """
        Parst Mana-Kosten und gibt ein Dictionary zurück
        Beispiel: "{2}{W}{U}" -> {'generic': 2, 'W': 1, 'U': 1}
        """
        if not mana_cost or pd.isna(mana_cost):
            return {'generic': 0, 'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0}
        
        result = {'generic': 0, 'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0}
        
        # Extrahiere alle Mana-Symbole
        symbols = re.findall(r'\{([^}]+)\}', mana_cost)
        
        for symbol in symbols:
            if symbol.isdigit():
                result['generic'] += int(symbol)
            elif symbol in ['W', 'U', 'B', 'R', 'G', 'C']:
                result[symbol] += 1
            elif '/' in symbol:  # Hybrid mana
                colors = symbol.split('/')
                for color in colors:
                    if color in ['W', 'U', 'B', 'R', 'G']:
                        result[color] += 0.5
            elif 'P' in symbol:  # Phyrexian mana
                color = symbol.replace('P', '')
                if color in ['W', 'U', 'B', 'R', 'G']:
                    result[color] += 1
                    
        return result
    
    def parse_json_field(self, field_value: Optional[str]) -> List[str]:
        """Parst JSON-String Felder aus der Datenbank"""
        if not field_value or pd.isna(field_value):
            return []
        try:
            return json.loads(field_value) if isinstance(field_value, str) else []
        except json.JSONDecodeError:
            return []
    
    def extract_card_features(self, card_row: pd.Series) -> np.ndarray:
        """
        Extrahiert Features aus einer Kartenzeile und erstellt einen Feature-Vektor
        """
        features = []
        
        # 1. Mana Cost Features (7 features: generic, W, U, B, R, G, C)
        mana_cost = self.parse_mana_cost(card_row.get('manaCost'))
        features.extend([
            mana_cost['generic'],
            mana_cost['W'],
            mana_cost['U'],
            mana_cost['B'],
            mana_cost['R'],
            mana_cost['G'],
            mana_cost['C']
        ])
        
        # 2. Mana Value / CMC (1 feature)
        features.append(card_row.get('manaValue', 0) if pd.notna(card_row.get('manaValue')) else 0)
        
        # 3. Color Identity (6 features: one-hot für W, U, B, R, G, colorless)
        color_identity = self.parse_json_field(card_row.get('colorIdentity'))
        color_vector = [0, 0, 0, 0, 0, 0]
        if not color_identity:
            color_vector[5] = 1  # Colorless
        else:
            for color in color_identity:
                if color in self.color_map:
                    color_vector[self.color_map[color]] = 1
        features.extend(color_vector)
        
        # 4. Card Types (Binary features für Creature, Instant, Sorcery, etc.)
        card_types = self.parse_json_field(card_row.get('types'))
        type_features = {
            'Creature': 0, 'Instant': 0, 'Sorcery': 0, 'Enchantment': 0,
            'Artifact': 0, 'Planeswalker': 0, 'Land': 0, 'Battle': 0
        }
        for card_type in card_types:
            if card_type in type_features:
                type_features[card_type] = 1
        features.extend(type_features.values())
        
        # 5. Power/Toughness (2 features, normalized)
        power = card_row.get('power', '0')
        toughness = card_row.get('toughness', '0')
        
        try:
            power_val = float(power) if power and power != '*' and power != 'X' else 0
        except (ValueError, TypeError):
            power_val = 0
            
        try:
            toughness_val = float(toughness) if toughness and toughness != '*' and toughness != 'X' else 0
        except (ValueError, TypeError):
            toughness_val = 0
            
        features.extend([power_val, toughness_val])
        
        # 6. Loyalty (1 feature, für Planeswalker)
        loyalty = card_row.get('loyalty', '0')
        try:
            loyalty_val = float(loyalty) if loyalty and loyalty != 'X' else 0
        except (ValueError, TypeError):
            loyalty_val = 0
        features.append(loyalty_val)
        
        # 7. Rarity (4 features: common, uncommon, rare, mythic)
        rarity = card_row.get('rarity', '').lower()
        rarity_vector = [
            1 if rarity == 'common' else 0,
            1 if rarity == 'uncommon' else 0,
            1 if rarity == 'rare' else 0,
            1 if rarity == 'mythic' else 0
        ]
        features.extend(rarity_vector)
        
        # 8. Keywords (wird später mit TF-IDF erweitert, vorerst count)
        keywords = self.parse_json_field(card_row.get('keywords'))
        features.append(len(keywords))
        
        # 9. Boolean Features
        features.extend([
            1 if card_row.get('isReserved') else 0,
            1 if card_row.get('hasAlternativeDeckLimit') else 0,
        ])
        
        return np.array(features, dtype=np.float32)
    
    def build_vocabularies(self, sample_size: int = 10000):
        """
        Baut Vokabulare für Keywords, Subtypes etc. auf
        Sample_size begrenzt die Anzahl der Karten für schnelleres Processing
        """
        conn = self.connect_db()
        
        # Sample von Karten holen
        query = f"""
            SELECT keywords, subtypes, types, text
            FROM cards 
            WHERE language = 'English'
            LIMIT {sample_size}
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Keywords sammeln
        for keywords_str in df['keywords'].dropna():
            keywords = self.parse_json_field(keywords_str)
            self.keyword_vocab.update(keywords)
        
        # Subtypes sammeln
        for subtypes_str in df['subtypes'].dropna():
            subtypes = self.parse_json_field(subtypes_str)
            self.subtype_vocab.update(subtypes)
        
        # Types sammeln
        for types_str in df['types'].dropna():
            types = self.parse_json_field(types_str)
            self.type_vocab.update(types)
        
        conn.close()
        
        print(f"Vokabular erstellt:")
        print(f"  - {len(self.keyword_vocab)} Keywords")
        print(f"  - {len(self.subtype_vocab)} Subtypes")
        print(f"  - {len(self.type_vocab)} Types")
    
    def get_feature_names(self) -> List[str]:
        """Gibt die Namen aller Features zurück"""
        names = [
            # Mana costs
            'mana_generic', 'mana_W', 'mana_U', 'mana_B', 'mana_R', 'mana_G', 'mana_C',
            # CMC
            'cmc',
            # Color identity
            'color_W', 'color_U', 'color_B', 'color_R', 'color_G', 'colorless',
            # Card types
            'type_creature', 'type_instant', 'type_sorcery', 'type_enchantment',
            'type_artifact', 'type_planeswalker', 'type_land', 'type_battle',
            # P/T/Loyalty
            'power', 'toughness', 'loyalty',
            # Rarity
            'rarity_common', 'rarity_uncommon', 'rarity_rare', 'rarity_mythic',
            # Other
            'keyword_count', 'is_reserved', 'has_alt_deck_limit'
        ]
        return names
    
    def process_all_cards(self, batch_size: int = 5000, limit: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Verarbeitet alle Karten und erstellt Embeddings
        
        Returns:
            embeddings: numpy array mit shape (n_cards, n_features)
            metadata: DataFrame mit Karteninformationen (uuid, name, setCode)
        """
        conn = self.connect_db()
        
        # Nur englische Karten, um Duplikate zu vermeiden
        count_query = "SELECT COUNT(*) FROM cards WHERE language = 'English'"
        if limit:
            count_query += f" LIMIT {limit}"
        
        total_cards = pd.read_sql_query(count_query, conn).iloc[0, 0]
        print(f"Verarbeite {total_cards} Karten...")
        
        all_embeddings = []
        all_metadata = []
        
        offset = 0
        while True:
            query = f"""
                SELECT * FROM cards 
                WHERE language = 'English'
                ORDER BY uuid
                LIMIT {batch_size} OFFSET {offset}
            """
            
            if limit and offset >= limit:
                break
                
            df_batch = pd.read_sql_query(query, conn)
            
            if len(df_batch) == 0:
                break
            
            # Embeddings für diesen Batch erstellen
            batch_embeddings = []
            batch_metadata = []
            
            for idx, row in df_batch.iterrows():
                embedding = self.extract_card_features(row)
                batch_embeddings.append(embedding)
                
                batch_metadata.append({
                    'uuid': row['uuid'],
                    'name': row['name'],
                    'setCode': row.get('setCode', ''),
                    'manaCost': row.get('manaCost', ''),
                    'type': row.get('type', '')
                })
            
            all_embeddings.extend(batch_embeddings)
            all_metadata.extend(batch_metadata)
            
            offset += batch_size
            print(f"Fortschritt: {offset}/{total_cards if not limit else min(limit, total_cards)} Karten verarbeitet")
            
            if limit and offset >= limit:
                break
        
        conn.close()
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        metadata_df = pd.DataFrame(all_metadata)
        
        print(f"\nIndexierung abgeschlossen!")
        print(f"Embedding Shape: {embeddings_array.shape}")
        print(f"Feature Dimensionen: {len(self.get_feature_names())}")
        
        return embeddings_array, metadata_df
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: pd.DataFrame, 
                       output_dir: str = 'data/embeddings'):
        """Speichert Embeddings und Metadata"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Embeddings als numpy array speichern
        embeddings_path = os.path.join(output_dir, 'card_embeddings.npy')
        np.save(embeddings_path, embeddings)
        print(f"Embeddings gespeichert: {embeddings_path}")
        
        # Metadata als CSV speichern
        metadata_path = os.path.join(output_dir, 'card_metadata.csv')
        metadata.to_csv(metadata_path, index=False)
        print(f"Metadata gespeichert: {metadata_path}")
        
        # Feature names speichern
        feature_names_path = os.path.join(output_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            for name in self.get_feature_names():
                f.write(f"{name}\n")
        print(f"Feature Names gespeichert: {feature_names_path}")
    
    def load_embeddings(self, input_dir: str = 'data/embeddings') -> Tuple[np.ndarray, pd.DataFrame]:
        """Lädt gespeicherte Embeddings und Metadata"""
        embeddings = np.load(os.path.join(input_dir, 'card_embeddings.npy'))
        metadata = pd.read_csv(os.path.join(input_dir, 'card_metadata.csv'))
        return embeddings, metadata


if __name__ == "__main__":
    # Beispiel-Nutzung
    embedder = MTGCardEmbedder()
    
    print("Erstelle Vokabulare...")
    embedder.build_vocabularies(sample_size=10000)
    
    print("\nVerarbeite Karten und erstelle Embeddings...")
    # Für Tests: nur 1000 Karten
    embeddings, metadata = embedder.process_all_cards(limit=1000)
    
    print("\nSpeichere Embeddings...")
    embedder.save_embeddings(embeddings, metadata)
    
    print("\nFertig! Die Embeddings sind bereit für den Denoising Autoencoder.")
