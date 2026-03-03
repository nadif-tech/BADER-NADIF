# =============================================================================
# IMPORTS (avec qrcode commenté)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import calendar
import hashlib
import sqlite3
import os
import sys
import logging
import traceback
import json
import time
import uuid
import re
import base64
import io
import csv
import zipfile
import shutil
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps, lru_cache
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import warnings
import random
import secrets
import threading
# import qrcode  # COMMENTÉ TEMPORAIREMENT
from PIL import Image, ImageDraw, ImageFont
import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gmao_enterprise.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GMAO Enterprise - Gestion de Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.gmao-enterprise.com/help',
        'Report a bug': 'https://www.gmao-enterprise.com/bug',
        'About': "# GMAO Enterprise\nVersion 3.0.0\n© 2024 Tous droits réservés"
    }
)

# =============================================================================
# ENUMS (inchangés)
# =============================================================================

class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    SUPERVISOR = "supervisor"
    TECHNICIAN = "technician"
    OPERATOR = "operator"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    ACCOUNTANT = "accountant"
    PURCHASER = "purchaser"
    STOCK_MANAGER = "stock_manager"

class AssetStatus(str, Enum):
    ACTIF = "Actif"
    EN_MAINTENANCE = "En maintenance"
    HORS_SERVICE = "Hors service"
    EN_RESERVE = "En réserve"
    EN_REPARATION = "En réparation"
    EN_INSTALLATION = "En installation"
    EN_TEST = "En test"
    EN_ATTENTE = "En attente"
    OBSOLETE = "Obsolète"
    VENDU = "Vendu"
    REBUT = "Rebut"

class InterventionStatus(str, Enum):
    OUVERTE = "Ouverte"
    ASSIGNEE = "Assignée"
    EN_COURS = "En cours"
    EN_PAUSE = "En pause"
    EN_ATTENTE_PIECES = "En attente de pièces"
    TERMINEE = "Terminée"
    FERMEE = "Fermée"
    ANNULEE = "Annulée"
    REPORTEE = "Reportée"
    A_VALIDER = "À valider"
    EN_CONTROLE = "En contrôle"

class PriorityLevel(str, Enum):
    TRES_BASSE = "Très basse"
    BASSE = "Basse"
    NORMALE = "Normale"
    HAUTE = "Haute"
    TRES_HAUTE = "Très haute"
    URGENTE = "Urgente"
    CRITIQUE = "Critique"

class MaintenanceType(str, Enum):
    PREVENTIVE = "Préventive"
    CURATIVE = "Curative"
    PREDICTIVE = "Prédictive"
    SYSTEMATIQUE = "Systématique"
    CONDITIONNELLE = "Conditionnelle"
    CORRECTIVE = "Corrective"
    AMELIORATIVE = "Améliorative"
    INSPECTION = "Inspection"
    CONTROLE = "Contrôle"
    CALIBRATION = "Calibration"

class StockMovementType(str, Enum):
    ENTREE = "Entrée"
    SORTIE = "Sortie"
    TRANSFERT = "Transfert"
    INVENTAIRE = "Inventaire"
    RETOUR = "Retour"
    REBUT = "Rebut"
    ADJUSTEMENT = "Ajustement"
    COMMANDE = "Commande"
    RECEPTION = "Réception"
    RESERVATION = "Réservation"

class DocumentType(str, Enum):
    FACTURE = "Facture"
    BON_LIVRAISON = "Bon de livraison"
    BON_COMMANDE = "Bon de commande"
    CONTRAT = "Contrat"
    MANUEL = "Manuel technique"
    FICHE_TECHNIQUE = "Fiche technique"
    CERTIFICAT = "Certificat"
    GARANTIE = "Garantie"
    PHOTO = "Photo"
    VIDEO = "Vidéo"
    RAPPORT = "Rapport"
    PROCEDURE = "Procédure"
    PLAN = "Plan"
    SCHEMA = "Schéma"

class NotificationType(str, Enum):
    MAINTENANCE = "maintenance"
    INTERVENTION = "intervention"
    STOCK = "stock"
    ALERTE = "alerte"
    SYSTEME = "système"
    TACHE = "tâche"
    RAPPORT = "rapport"
    DOCUMENT = "document"
    VALIDATION = "validation"
    APPROBATION = "approbation"

class ReportFormat(str, Enum):
    PDF = "PDF"
    EXCEL = "Excel"
    CSV = "CSV"
    HTML = "HTML"
    JSON = "JSON"
    XML = "XML"
    WORD = "Word"
    POWERPOINT = "PowerPoint"
    IMAGE = "Image"
    MARKDOWN = "Markdown"

class UnitType(str, Enum):
    PIECE = "pièce"
    METRE = "mètre"
    KILOGRAMME = "kilogramme"
    LITRE = "litre"
    HEURE = "heure"
    JOUR = "jour"
    MOIS = "mois"
    ANNEE = "année"
    SERVICE = "service"
    FORFAIT = "forfait"

class ContractType(str, Enum):
    MAINTENANCE = "maintenance"
    LOCATION = "location"
    PRESTATION = "prestation"
    FOURNITURE = "fourniture"
    SERVICE = "service"
    GARANTIE = "garantie"
    ASSURANCE = "assurance"

class PaymentMethod(str, Enum):
    CARTE_BANCAIRE = "Carte bancaire"
    VIREMENT = "Virement"
    CHEQUE = "Chèque"
    ESPECES = "Espèces"
    PRELEVEMENT = "Prélèvement"
    LCR = "LCR"
    TRAITE = "Traite"
    LETTRE_CHARGE = "Lettre de change"

# =============================================================================
# DATA CLASSES (inchangées)
# =============================================================================

@dataclass
class User:
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    first_name: str = ""
    last_name: str = ""
    role: UserRole = UserRole.VIEWER
    department: str = ""
    position: str = ""
    phone: str = ""
    mobile: str = ""
    address: str = ""
    city: str = ""
    postal_code: str = ""
    country: str = "France"
    hire_date: Optional[date] = None
    birth_date: Optional[date] = None
    emergency_contact: str = ""
    emergency_phone: str = ""
    photo: Optional[str] = None
    signature: Optional[str] = None
    notes: str = ""
    is_active: bool = True
    is_deleted: bool = False
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def initials(self) -> str:
        return f"{self.first_name[0] if self.first_name else ''}{self.last_name[0] if self.last_name else ''}".upper()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Asset:
    id: Optional[int] = None
    code: str = ""
    name: str = ""
    type: str = ""
    model: str = ""
    manufacturer: str = ""
    serial_number: str = ""
    barcode: str = ""
    qr_code: str = ""
    rfid_tag: str = ""
    acquisition_date: Optional[date] = None
    commissioning_date: Optional[date] = None
    warranty_end_date: Optional[date] = None
    warranty_days: int = 0
    location: str = ""
    department: str = ""
    building: str = ""
    floor: str = ""
    room: str = ""
    responsible_id: Optional[int] = None
    status: AssetStatus = AssetStatus.ACTIF
    purchase_price: float = 0.0
    current_value: float = 0.0
    depreciation_rate: float = 0.0
    useful_life_years: int = 0
    last_maintenance_date: Optional[date] = None
    next_maintenance_date: Optional[date] = None
    maintenance_frequency_days: int = 0
    meter_type: str = ""
    current_meter_value: float = 0.0
    meter_unit: str = ""
    meter_reset_date: Optional[date] = None
    supplier_id: Optional[int] = None
    manufacturer_id: Optional[int] = None
    category_id: Optional[int] = None
    subcategory_id: Optional[int] = None
    criticality: str = "Normal"
    energy_consumption: float = 0.0
    co2_emission: float = 0.0
    documentation: str = ""
    photo: Optional[str] = None
    technical_sheet: Optional[str] = None
    notes: str = ""
    is_active: bool = True
    is_deleted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def age_days(self) -> int:
        if self.commissioning_date:
            return (date.today() - self.commissioning_date).days
        return 0

    @property
    def age_years(self) -> float:
        return self.age_days / 365.25

    @property
    def warranty_status(self) -> str:
        if not self.warranty_end_date:
            return "Sans garantie"
        if self.warranty_end_date >= date.today():
            days_left = (self.warranty_end_date - date.today()).days
            return f"Sous garantie ({days_left} jours)"
        return "Garantie expirée"

@dataclass
class Intervention:
    id: Optional[int] = None
    number: str = ""
    title: str = ""
    description: str = ""
    type: str = ""
    priority: PriorityLevel = PriorityLevel.NORMALE
    status: InterventionStatus = InterventionStatus.OUVERTE
    asset_id: int = 0
    asset_code: str = ""
    asset_name: str = ""
    requester_id: Optional[int] = None
    requester_name: str = ""
    technician_id: Optional[int] = None
    technician_name: str = ""
    supervisor_id: Optional[int] = None
    supervisor_name: str = ""
    opening_date: datetime = field(default_factory=datetime.now)
    assignment_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    pause_date: Optional[datetime] = None
    resume_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    closing_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    downtime_hours: float = 0.0
    cause: str = ""
    solution: str = ""
    observations: str = ""
    work_performed: str = ""
    parts_used: str = ""
    parts_cost: float = 0.0
    labor_cost: float = 0.0
    travel_cost: float = 0.0
    other_cost: float = 0.0
    total_cost: float = 0.0
    satisfaction_score: Optional[int] = None
    satisfaction_comment: str = ""
    requires_followup: bool = False
    followup_id: Optional[int] = None
    is_urgent: bool = False
    is_planned: bool = False
    is_preventive: bool = False
    is_corrective: bool = False
    is_warranty: bool = False
    is_billed: bool = False
    invoice_number: str = ""
    invoice_date: Optional[date] = None
    invoice_amount: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def duration_hours(self) -> float:
        if self.start_date and self.completion_date:
            delta = self.completion_date - self.start_date
            return delta.total_seconds() / 3600
        return 0.0

    @property
    def response_time_hours(self) -> float:
        if self.opening_date and self.start_date:
            delta = self.start_date - self.opening_date
            return delta.total_seconds() / 3600
        return 0.0

    @property
    def resolution_time_hours(self) -> float:
        if self.opening_date and self.closing_date:
            delta = self.closing_date - self.opening_date
            return delta.total_seconds() / 3600
        return 0.0

@dataclass
class SparePart:
    id: Optional[int] = None
    code: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    subcategory: str = ""
    brand: str = ""
    model: str = ""
    supplier_id: Optional[int] = None
    supplier_code: str = ""
    manufacturer_id: Optional[int] = None
    manufacturer_code: str = ""
    barcode: str = ""
    qr_code: str = ""
    rfid_tag: str = ""
    unit: UnitType = UnitType.PIECE
    unit_price: float = 0.0
    purchase_price: float = 0.0
    selling_price: float = 0.0
    vat_rate: float = 20.0
    quantity: int = 0
    min_quantity: int = 0
    max_quantity: int = 100
    reorder_point: int = 0
    reorder_quantity: int = 0
    location: str = ""
    warehouse: str = ""
    aisle: str = ""
    rack: str = ""
    bin: str = ""
    stock_value: float = 0.0
    last_purchase_date: Optional[date] = None
    last_sale_date: Optional[date] = None
    last_inventory_date: Optional[date] = None
    is_active: bool = True
    is_deleted: bool = False
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def is_low_stock(self) -> bool:
        return self.quantity <= self.min_quantity

    @property
    def is_out_of_stock(self) -> bool:
        return self.quantity <= 0

    @property
    def needs_reorder(self) -> bool:
        return self.quantity <= self.reorder_point

    @property
    def stock_status(self) -> str:
        if self.quantity <= 0:
            return "Rupture"
        elif self.quantity <= self.min_quantity:
            return "Critique"
        elif self.quantity <= self.reorder_point:
            return "Faible"
        elif self.quantity >= self.max_quantity:
            return "Surcharge"
        else:
            return "Normal"

@dataclass
class Supplier:
    id: Optional[int] = None
    code: str = ""
    name: str = ""
    legal_name: str = ""
    type: str = ""
    category: str = ""
    siret: str = ""
    siren: str = ""
    vat_number: str = ""
    website: str = ""
    email: str = ""
    phone: str = ""
    fax: str = ""
    mobile: str = ""
    address: str = ""
    address2: str = ""
    postal_code: str = ""
    city: str = ""
    state: str = ""
    country: str = "France"
    contact_first_name: str = ""
    contact_last_name: str = ""
    contact_position: str = ""
    contact_phone: str = ""
    contact_mobile: str = ""
    contact_email: str = ""
    payment_terms: str = ""
    delivery_terms: str = ""
    delivery_delay_days: int = 0
    minimum_order: float = 0.0
    currency: str = "EUR"
    bank_name: str = ""
    bank_account: str = ""
    iban: str = ""
    bic: str = ""
    rating: int = 0
    notes: str = ""
    is_active: bool = True
    is_deleted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def full_address(self) -> str:
        parts = []
        if self.address:
            parts.append(self.address)
        if self.address2:
            parts.append(self.address2)
        if self.postal_code:
            parts.append(self.postal_code)
        if self.city:
            parts.append(self.city)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    @property
    def contact_full_name(self) -> str:
        return f"{self.contact_first_name} {self.contact_last_name}".strip()

@dataclass
class Document:
    id: Optional[int] = None
    number: str = ""
    title: str = ""
    description: str = ""
    type: DocumentType = DocumentType.DOCUMENT
    category: str = ""
    entity_type: str = ""
    entity_id: int = 0
    filename: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: str = ""
    file_hash: str = ""
    version: str = "1.0"
    is_public: bool = False
    is_archived: bool = False
    is_deleted: bool = False
    expiry_date: Optional[date] = None
    tags: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def size_human(self) -> str:
        """Convertit la taille du fichier en format lisible sans humanize"""
        size = self.file_size
        for unit in ['o', 'Ko', 'Mo', 'Go']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} To"

@dataclass
class Notification:
    id: Optional[int] = None
    user_id: int = 0
    type: NotificationType = NotificationType.SYSTEME
    title: str = ""
    message: str = ""
    link: str = ""
    is_read: bool = False
    is_archived: bool = False
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Contract:
    id: Optional[int] = None
    number: str = ""
    title: str = ""
    description: str = ""
    type: ContractType = ContractType.MAINTENANCE
    supplier_id: int = 0
    client_id: Optional[int] = None
    start_date: date = field(default_factory=date.today)
    end_date: Optional[date] = None
    renewal_date: Optional[date] = None
    is_auto_renew: bool = False
    amount: float = 0.0
    currency: str = "EUR"
    payment_method: PaymentMethod = PaymentMethod.VIREMENT
    payment_terms: str = ""
    payment_frequency: str = "Mensuel"
    documents: str = ""
    notes: str = ""
    is_active: bool = True
    is_deleted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

    @property
    def days_until_expiry(self) -> int:
        if self.end_date:
            return (self.end_date - date.today()).days
        return 0

    @property
    def is_expiring_soon(self) -> bool:
        return 0 < self.days_until_expiry <= 30

    @property
    def is_expired(self) -> bool:
        return self.end_date and self.end_date < date.today()

@dataclass
class MeterReading:
    id: Optional[int] = None
    asset_id: int = 0
    meter_type: str = ""
    previous_value: float = 0.0
    current_value: float = 0.0
    difference: float = 0.0
    reading_date: datetime = field(default_factory=datetime.now)
    reading_method: str = "Manuel"
    reader_id: Optional[int] = None
    notes: str = ""
    is_verified: bool = False
    verified_by: Optional[int] = None
    verified_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def daily_average(self) -> float:
        if self.reading_date and hasattr(self, 'previous_date'):
            days = (self.reading_date - self.previous_date).days
            if days > 0:
                return self.difference / days
        return 0.0

@dataclass
class WorkOrder:
    id: Optional[int] = None
    number: str = ""
    title: str = ""
    description: str = ""
    intervention_id: Optional[int] = None
    asset_id: int = 0
    technician_id: int = 0
    supervisor_id: Optional[int] = None
    priority: PriorityLevel = PriorityLevel.NORMALE
    status: str = "Planifié"
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    parts_used: str = ""
    parts_cost: float = 0.0
    labor_cost: float = 0.0
    total_cost: float = 0.0
    instructions: str = ""
    safety_instructions: str = ""
    tools_required: str = ""
    is_completed: bool = False
    completion_notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PurchaseOrder:
    id: Optional[int] = None
    number: str = ""
    supplier_id: int = 0
    order_date: date = field(default_factory=date.today)
    expected_delivery_date: Optional[date] = None
    actual_delivery_date: Optional[date] = None
    status: str = "Brouillon"
    items: str = ""
    subtotal: float = 0.0
    tax_amount: float = 0.0
    shipping_cost: float = 0.0
    total_amount: float = 0.0
    currency: str = "EUR"
    payment_terms: str = ""
    shipping_address: str = ""
    billing_address: str = ""
    notes: str = ""
    is_approved: bool = False
    approved_by: Optional[int] = None
    approved_at: Optional[datetime] = None
    is_received: bool = False
    received_by: Optional[int] = None
    received_at: Optional[datetime] = None
    is_invoiced: bool = False
    invoice_number: str = ""
    invoice_date: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[int] = None
    updated_by: Optional[int] = None

# =============================================================================
# EXCEPTIONS (inchangées)
# =============================================================================

class GMAOException(Exception):
    def __init__(self, message: str = "", code: int = 500, details: Any = None):
        self.message = message or "Une erreur est survenue"
        self.code = code
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details
        }

class AuthenticationError(GMAOException):
    def __init__(self, message: str = "Nom d'utilisateur ou mot de passe incorrect", details: Any = None):
        super().__init__(message, 401, details)

class AuthorizationError(GMAOException):
    def __init__(self, message: str = "Accès non autorisé", details: Any = None):
        super().__init__(message, 403, details)

class ValidationError(GMAOException):
    def __init__(self, message: str = "Données invalides", details: Any = None):
        super().__init__(message, 400, details)

class NotFoundError(GMAOException):
    def __init__(self, message: str = "Ressource non trouvée", details: Any = None):
        super().__init__(message, 404, details)

class DuplicateEntryError(GMAOException):
    def __init__(self, message: str = "Cette entrée existe déjà", details: Any = None):
        super().__init__(message, 409, details)

class DatabaseError(GMAOException):
    def __init__(self, message: str = "Erreur de base de données", details: Any = None):
        super().__init__(message, 500, details)

class BusinessRuleError(GMAOException):
    def __init__(self, message: str = "Règle métier non respectée", details: Any = None):
        super().__init__(message, 422, details)

class ConfigurationError(GMAOException):
    def __init__(self, message: str = "Erreur de configuration", details: Any = None):
        super().__init__(message, 500, details)

class IntegrationError(GMAOException):
    def __init__(self, message: str = "Erreur d'intégration", details: Any = None):
        super().__init__(message, 500, details)

class FileError(GMAOException):
    def __init__(self, message: str = "Erreur de fichier", details: Any = None):
        super().__init__(message, 500, details)

class NetworkError(GMAOException):
    def __init__(self, message: str = "Erreur réseau", details: Any = None):
        super().__init__(message, 503, details)

class TimeoutError(GMAOException):
    def __init__(self, message: str = "Délai d'attente dépassé", details: Any = None):
        super().__init__(message, 408, details)

class QuotaExceededError(GMAOException):
    def __init__(self, message: str = "Quota dépassé", details: Any = None):
        super().__init__(message, 429, details)

class MaintenanceModeError(GMAOException):
    def __init__(self, message: str = "Application en mode maintenance", details: Any = None):
        super().__init__(message, 503, details)

# =============================================================================
# DATABASE MANAGER (corrigé)
# =============================================================================

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool = []
        self.pool_size = 10
        self.lock = threading.Lock()
        self.setup_database()

    def setup_database(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = 10000")
                cursor.execute("PRAGMA temp_store = MEMORY")
                cursor.execute("PRAGMA mmap_size = 30000000000")
                self.create_tables(cursor)
                self.create_indexes(cursor)
                self.create_triggers(cursor)
                self.create_views(cursor)
                self.insert_default_data(cursor)
                conn.commit()
            logger.info("Base de données initialisée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise DatabaseError(f"Erreur d'initialisation: {e}")

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            with self.lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
            if not conn:
                conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Erreur SQLite: {e}")
            raise DatabaseError(f"Erreur de base de données: {e}")
        finally:
            if conn:
                with self.lock:
                    if len(self.connection_pool) < self.pool_size:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        with self.get_connection() as conn:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            return df

    def execute_insert(self, query: str, params: tuple = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid

    def execute_update(self, query: str, params: tuple = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount

    def execute_delete(self, query: str, params: tuple = None) -> int:
        return self.execute_update(query, params)

    def execute_many(self, query: str, params_list: list) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def execute_script(self, script: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(script)

    def transaction(self):
        return self.get_connection()

    def create_tables(self, cursor):
        # Table des utilisateurs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                department TEXT,
                position TEXT,
                phone TEXT,
                mobile TEXT,
                address TEXT,
                city TEXT,
                postal_code TEXT,
                country TEXT DEFAULT 'France',
                hire_date DATE,
                birth_date DATE,
                emergency_contact TEXT,
                emergency_phone TEXT,
                photo TEXT,
                signature TEXT,
                notes TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_deleted BOOLEAN DEFAULT 0,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des équipements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                model TEXT,
                manufacturer TEXT,
                serial_number TEXT,
                barcode TEXT UNIQUE,
                qr_code TEXT UNIQUE,
                rfid_tag TEXT UNIQUE,
                acquisition_date DATE,
                commissioning_date DATE,
                warranty_end_date DATE,
                warranty_days INTEGER DEFAULT 0,
                location TEXT,
                department TEXT,
                building TEXT,
                floor TEXT,
                room TEXT,
                responsible_id INTEGER,
                status TEXT DEFAULT 'Actif',
                purchase_price REAL DEFAULT 0,
                current_value REAL DEFAULT 0,
                depreciation_rate REAL DEFAULT 0,
                useful_life_years INTEGER DEFAULT 0,
                last_maintenance_date DATE,
                next_maintenance_date DATE,
                maintenance_frequency_days INTEGER DEFAULT 0,
                meter_type TEXT,
                current_meter_value REAL DEFAULT 0,
                meter_unit TEXT,
                meter_reset_date DATE,
                supplier_id INTEGER,
                manufacturer_id INTEGER,
                category_id INTEGER,
                subcategory_id INTEGER,
                criticality TEXT DEFAULT 'Normal',
                energy_consumption REAL DEFAULT 0,
                co2_emission REAL DEFAULT 0,
                documentation TEXT,
                photo TEXT,
                technical_sheet TEXT,
                notes TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_deleted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (responsible_id) REFERENCES users(id),
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (manufacturer_id) REFERENCES suppliers(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des catégories d'équipements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                parent_id INTEGER,
                level INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES asset_categories(id)
            )
        """)

        # Table des interventions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                priority TEXT DEFAULT 'Normale',
                status TEXT DEFAULT 'Ouverte',
                asset_id INTEGER NOT NULL,
                requester_id INTEGER,
                technician_id INTEGER,
                supervisor_id INTEGER,
                opening_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assignment_date TIMESTAMP,
                start_date TIMESTAMP,
                pause_date TIMESTAMP,
                resume_date TIMESTAMP,
                completion_date TIMESTAMP,
                closing_date TIMESTAMP,
                due_date TIMESTAMP,
                estimated_duration REAL DEFAULT 0,
                actual_duration REAL DEFAULT 0,
                downtime_hours REAL DEFAULT 0,
                cause TEXT,
                solution TEXT,
                observations TEXT,
                work_performed TEXT,
                parts_used TEXT,
                parts_cost REAL DEFAULT 0,
                labor_cost REAL DEFAULT 0,
                travel_cost REAL DEFAULT 0,
                other_cost REAL DEFAULT 0,
                total_cost REAL DEFAULT 0,
                satisfaction_score INTEGER,
                satisfaction_comment TEXT,
                requires_followup BOOLEAN DEFAULT 0,
                followup_id INTEGER,
                is_urgent BOOLEAN DEFAULT 0,
                is_planned BOOLEAN DEFAULT 0,
                is_preventive BOOLEAN DEFAULT 0,
                is_corrective BOOLEAN DEFAULT 0,
                is_warranty BOOLEAN DEFAULT 0,
                is_billed BOOLEAN DEFAULT 0,
                invoice_number TEXT,
                invoice_date DATE,
                invoice_amount REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
                FOREIGN KEY (requester_id) REFERENCES users(id),
                FOREIGN KEY (technician_id) REFERENCES users(id),
                FOREIGN KEY (supervisor_id) REFERENCES users(id),
                FOREIGN KEY (followup_id) REFERENCES interventions(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des pièces détachées
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spare_parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                subcategory TEXT,
                brand TEXT,
                model TEXT,
                supplier_id INTEGER,
                supplier_code TEXT,
                manufacturer_id INTEGER,
                manufacturer_code TEXT,
                barcode TEXT UNIQUE,
                qr_code TEXT UNIQUE,
                rfid_tag TEXT UNIQUE,
                unit TEXT DEFAULT 'pièce',
                unit_price REAL DEFAULT 0,
                purchase_price REAL DEFAULT 0,
                selling_price REAL DEFAULT 0,
                vat_rate REAL DEFAULT 20,
                quantity INTEGER DEFAULT 0,
                min_quantity INTEGER DEFAULT 0,
                max_quantity INTEGER DEFAULT 100,
                reorder_point INTEGER DEFAULT 0,
                reorder_quantity INTEGER DEFAULT 0,
                location TEXT,
                warehouse TEXT,
                aisle TEXT,
                rack TEXT,
                bin TEXT,
                stock_value REAL DEFAULT 0,
                last_purchase_date DATE,
                last_sale_date DATE,
                last_inventory_date DATE,
                is_active BOOLEAN DEFAULT 1,
                is_deleted BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (manufacturer_id) REFERENCES suppliers(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des fournisseurs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suppliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                legal_name TEXT,
                type TEXT,
                category TEXT,
                siret TEXT,
                siren TEXT,
                vat_number TEXT,
                website TEXT,
                email TEXT,
                phone TEXT,
                fax TEXT,
                mobile TEXT,
                address TEXT,
                address2 TEXT,
                postal_code TEXT,
                city TEXT,
                state TEXT,
                country TEXT DEFAULT 'France',
                contact_first_name TEXT,
                contact_last_name TEXT,
                contact_position TEXT,
                contact_phone TEXT,
                contact_mobile TEXT,
                contact_email TEXT,
                payment_terms TEXT,
                delivery_terms TEXT,
                delivery_delay_days INTEGER DEFAULT 0,
                minimum_order REAL DEFAULT 0,
                currency TEXT DEFAULT 'EUR',
                bank_name TEXT,
                bank_account TEXT,
                iban TEXT,
                bic TEXT,
                rating INTEGER DEFAULT 0,
                notes TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_deleted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des mouvements de stock
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                before_quantity INTEGER,
                after_quantity INTEGER,
                unit_price REAL,
                total_price REAL,
                reference_type TEXT,
                reference_id INTEGER,
                document_number TEXT,
                movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT,
                notes TEXT,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (part_id) REFERENCES spare_parts(id) ON DELETE CASCADE,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Table des documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                category TEXT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                file_hash TEXT,
                version TEXT DEFAULT '1.0',
                is_public BOOLEAN DEFAULT 0,
                is_archived BOOLEAN DEFAULT 0,
                is_deleted BOOLEAN DEFAULT 0,
                expiry_date DATE,
                tags TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des notifications
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT,
                link TEXT,
                is_read BOOLEAN DEFAULT 0,
                is_archived BOOLEAN DEFAULT 0,
                read_at TIMESTAMP,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Table des contrats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                supplier_id INTEGER NOT NULL,
                client_id INTEGER,
                start_date DATE NOT NULL,
                end_date DATE,
                renewal_date DATE,
                is_auto_renew BOOLEAN DEFAULT 0,
                amount REAL DEFAULT 0,
                currency TEXT DEFAULT 'EUR',
                payment_method TEXT DEFAULT 'Virement',
                payment_terms TEXT,
                payment_frequency TEXT DEFAULT 'Mensuel',
                documents TEXT,
                notes TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_deleted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (client_id) REFERENCES users(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des relevés de compteurs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meter_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id INTEGER NOT NULL,
                meter_type TEXT NOT NULL,
                previous_value REAL,
                current_value REAL NOT NULL,
                difference REAL,
                reading_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reading_method TEXT DEFAULT 'Manuel',
                reader_id INTEGER,
                notes TEXT,
                is_verified BOOLEAN DEFAULT 0,
                verified_by INTEGER,
                verified_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
                FOREIGN KEY (reader_id) REFERENCES users(id),
                FOREIGN KEY (verified_by) REFERENCES users(id)
            )
        """)

        # Table des ordres de travail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                intervention_id INTEGER,
                asset_id INTEGER NOT NULL,
                technician_id INTEGER NOT NULL,
                supervisor_id INTEGER,
                priority TEXT DEFAULT 'Normale',
                status TEXT DEFAULT 'Planifié',
                planned_start TIMESTAMP,
                planned_end TIMESTAMP,
                actual_start TIMESTAMP,
                actual_end TIMESTAMP,
                estimated_hours REAL DEFAULT 0,
                actual_hours REAL DEFAULT 0,
                parts_used TEXT,
                parts_cost REAL DEFAULT 0,
                labor_cost REAL DEFAULT 0,
                total_cost REAL DEFAULT 0,
                instructions TEXT,
                safety_instructions TEXT,
                tools_required TEXT,
                is_completed BOOLEAN DEFAULT 0,
                completion_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (intervention_id) REFERENCES interventions(id),
                FOREIGN KEY (asset_id) REFERENCES assets(id),
                FOREIGN KEY (technician_id) REFERENCES users(id),
                FOREIGN KEY (supervisor_id) REFERENCES users(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des bons de commande
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS purchase_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT UNIQUE NOT NULL,
                supplier_id INTEGER NOT NULL,
                order_date DATE DEFAULT CURRENT_DATE,
                expected_delivery_date DATE,
                actual_delivery_date DATE,
                status TEXT DEFAULT 'Brouillon',
                items TEXT,
                subtotal REAL DEFAULT 0,
                tax_amount REAL DEFAULT 0,
                shipping_cost REAL DEFAULT 0,
                total_amount REAL DEFAULT 0,
                currency TEXT DEFAULT 'EUR',
                payment_terms TEXT,
                shipping_address TEXT,
                billing_address TEXT,
                notes TEXT,
                is_approved BOOLEAN DEFAULT 0,
                approved_by INTEGER,
                approved_at TIMESTAMP,
                is_received BOOLEAN DEFAULT 0,
                received_by INTEGER,
                received_at TIMESTAMP,
                is_invoiced BOOLEAN DEFAULT 0,
                invoice_number TEXT,
                invoice_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (approved_by) REFERENCES users(id),
                FOREIGN KEY (received_by) REFERENCES users(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des historiques
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS histories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                user_id INTEGER,
                old_values TEXT,
                new_values TEXT,
                changes TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Table des paramètres
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT,
                type TEXT DEFAULT 'string',
                category TEXT,
                description TEXT,
                is_public BOOLEAN DEFAULT 0,
                is_system BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table des préférences utilisateur
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                theme TEXT DEFAULT 'light',
                language TEXT DEFAULT 'fr',
                notifications_enabled BOOLEAN DEFAULT 1,
                email_notifications BOOLEAN DEFAULT 1,
                dashboard_layout TEXT,
                favorites TEXT,
                recent_items TEXT,
                filters TEXT,
                columns_visibility TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Table des sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_activity TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Table des logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                logger TEXT,
                message TEXT,
                module TEXT,
                function TEXT,
                line INTEGER,
                user_id INTEGER,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Table des tâches planifiées
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                schedule TEXT NOT NULL,
                parameters TEXT,
                last_run TIMESTAMP,
                next_run TIMESTAMP,
                status TEXT DEFAULT 'active',
                is_running BOOLEAN DEFAULT 0,
                last_result TEXT,
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des rapports
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                format TEXT DEFAULT 'PDF',
                query TEXT,
                parameters TEXT,
                template TEXT,
                output_path TEXT,
                schedule_id INTEGER,
                is_scheduled BOOLEAN DEFAULT 0,
                last_generated TIMESTAMP,
                next_generation TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (schedule_id) REFERENCES scheduled_tasks(id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des dashboards
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                layout TEXT,
                widgets TEXT,
                is_default BOOLEAN DEFAULT 0,
                is_public BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des widgets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS widgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dashboard_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                position INTEGER,
                size TEXT,
                configuration TEXT,
                data_source TEXT,
                refresh_interval INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dashboard_id) REFERENCES dashboards(id) ON DELETE CASCADE
            )
        """)

        # Table des alertes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                condition TEXT NOT NULL,
                threshold TEXT,
                severity TEXT DEFAULT 'warning',
                notification_type TEXT,
                notification_target TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_triggered TIMESTAMP,
                trigger_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                updated_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id),
                FOREIGN KEY (updated_by) REFERENCES users(id)
            )
        """)

        # Table des audits
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id INTEGER,
                old_value TEXT,
                new_value TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Table des exports
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                format TEXT NOT NULL,
                query TEXT,
                parameters TEXT,
                file_path TEXT,
                file_size INTEGER,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Table des imports
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                format TEXT NOT NULL,
                file_path TEXT,
                mapping TEXT,
                status TEXT DEFAULT 'pending',
                imported_records INTEGER DEFAULT 0,
                error_records INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Table des API tokens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                permissions TEXT,
                last_used TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Table des webhooks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS webhooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                events TEXT,
                secret TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_triggered TIMESTAMP,
                trigger_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Table des intégrations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                configuration TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_sync TIMESTAMP,
                sync_status TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

    def create_indexes(self, cursor):
        # Index pour la table users
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)")

        # Index pour la table assets
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_code ON assets(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_responsible ON assets(responsible_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_location ON assets(location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_next_maintenance ON assets(next_maintenance_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_serial ON assets(serial_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_barcode ON assets(barcode)")

        # Index pour la table interventions
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_number ON interventions(number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_asset ON interventions(asset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_technician ON interventions(technician_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_status ON interventions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_priority ON interventions(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_opening_date ON interventions(opening_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_closing_date ON interventions(closing_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interventions_requester ON interventions(requester_id)")

        # Index pour la table spare_parts
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_code ON spare_parts(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_name ON spare_parts(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_category ON spare_parts(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_supplier ON spare_parts(supplier_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_quantity ON spare_parts(quantity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_location ON spare_parts(location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_barcode ON spare_parts(barcode)")

        # Index pour la table suppliers
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_code ON suppliers(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_name ON suppliers(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_city ON suppliers(city)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_country ON suppliers(country)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suppliers_rating ON suppliers(rating)")

        # Index pour la table stock_movements
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_part ON stock_movements(part_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_type ON stock_movements(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_date ON stock_movements(movement_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_reference ON stock_movements(reference_type, reference_id)")

        # Index pour la table documents
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_entity ON documents(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_expiry ON documents(expiry_date)")

        # Index pour la table notifications
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at)")

        # Index pour la table histories
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_histories_entity ON histories(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_histories_user ON histories(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_histories_created ON histories(created_at)")

        # Index pour la table logs
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_created ON logs(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_user ON logs(user_id)")

        # Index pour la table audits
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audits_user ON audits(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audits_action ON audits(action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audits_entity ON audits(entity_type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audits_created ON audits(created_at)")

    def create_triggers(self, cursor):
        # Trigger pour updated_at sur users
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_users_updated_at 
            AFTER UPDATE ON users
            BEGIN
                UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)

        # Trigger pour updated_at sur assets
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_assets_updated_at 
            AFTER UPDATE ON assets
            BEGIN
                UPDATE assets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)

        # Trigger pour updated_at sur interventions
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_interventions_updated_at 
            AFTER UPDATE ON interventions
            BEGIN
                UPDATE interventions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)

        # Trigger pour updated_at sur spare_parts
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_parts_updated_at 
            AFTER UPDATE ON spare_parts
            BEGIN
                UPDATE spare_parts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)

        # Trigger pour updated_at sur suppliers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_suppliers_updated_at 
            AFTER UPDATE ON suppliers
            BEGIN
                UPDATE suppliers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)

        # Trigger pour la mise à jour du stock après mouvement
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_stock_after_insert 
            AFTER INSERT ON stock_movements
            BEGIN
                UPDATE spare_parts 
                SET quantity = quantity + NEW.quantity,
                    stock_value = (quantity + NEW.quantity) * unit_price,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.part_id;
            END;
        """)

        # Trigger pour l'historique automatique
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_interventions_history 
            AFTER UPDATE ON interventions
            BEGIN
                INSERT INTO histories (entity_type, entity_id, action, user_id, old_values, new_values, created_at)
                VALUES ('intervention', NEW.id, 'UPDATE', NEW.updated_by, OLD, NEW, CURRENT_TIMESTAMP);
            END;
        """)

        # Trigger pour la validation des dates
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_validate_intervention_dates 
            BEFORE INSERT ON interventions
            BEGIN
                SELECT CASE
                    WHEN NEW.closing_date < NEW.opening_date THEN
                        RAISE (ABORT, 'La date de clôture ne peut pas être antérieure à la date d\'ouverture')
                END;
            END;
        """)

        # Trigger pour le calcul automatique du total d'une intervention
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_intervention_total_cost 
            BEFORE UPDATE ON interventions
            BEGIN
                UPDATE interventions 
                SET total_cost = NEW.parts_cost + NEW.labor_cost + NEW.travel_cost + NEW.other_cost
                WHERE id = NEW.id;
            END;
        """)

        # Trigger pour le calcul automatique de la différence de compteur
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_meter_difference 
            BEFORE INSERT ON meter_readings
            BEGIN
                SELECT CASE
                    WHEN NEW.previous_value IS NOT NULL THEN
                        NEW.difference = NEW.current_value - NEW.previous_value
                END;
            END;
        """)

        # Trigger pour la mise à jour du compteur de l'équipement
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trigger_update_asset_meter 
            AFTER INSERT ON meter_readings
            BEGIN
                UPDATE assets 
                SET current_meter_value = NEW.current_value,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.asset_id;
            END;
        """)

    def create_views(self, cursor):
        # Vue des interventions avec détails
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_interventions_details AS
            SELECT 
                i.*,
                a.code AS asset_code,
                a.name AS asset_name,
                a.location AS asset_location,
                r.first_name || ' ' || r.last_name AS requester_name,
                t.first_name || ' ' || t.last_name AS technician_name,
                s.first_name || ' ' || s.last_name AS supervisor_name,
                julianday(COALESCE(i.closing_date, 'now')) - julianday(i.opening_date) AS resolution_days,
                CASE 
                    WHEN i.closing_date IS NOT NULL AND i.opening_date IS NOT NULL 
                    THEN (julianday(i.closing_date) - julianday(i.opening_date)) * 24 
                    ELSE NULL 
                END AS resolution_hours
            FROM interventions i
            LEFT JOIN assets a ON i.asset_id = a.id
            LEFT JOIN users r ON i.requester_id = r.id
            LEFT JOIN users t ON i.technician_id = t.id
            LEFT JOIN users s ON i.supervisor_id = s.id
        """)

        # Vue des équipements avec statistiques
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_assets_stats AS
            SELECT 
                a.*,
                COUNT(DISTINCT i.id) AS total_interventions,
                SUM(CASE WHEN i.status = 'Terminée' THEN 1 ELSE 0 END) AS completed_interventions,
                SUM(CASE WHEN i.status = 'En cours' THEN 1 ELSE 0 END) AS ongoing_interventions,
                SUM(CASE WHEN i.priority = 'Urgente' AND i.status != 'Terminée' THEN 1 ELSE 0 END) AS urgent_interventions,
                AVG(i.satisfaction_score) AS avg_satisfaction,
                SUM(i.total_cost) AS total_maintenance_cost,
                MAX(i.completion_date) AS last_intervention_date,
                MIN(i.opening_date) AS first_intervention_date,
                julianday('now') - julianday(COALESCE(a.last_maintenance_date, a.commissioning_date, 'now')) AS days_since_last_maintenance
            FROM assets a
            LEFT JOIN interventions i ON a.id = i.asset_id
            GROUP BY a.id
        """)

        # Vue du stock avec alertes
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_stock_alerts AS
            SELECT 
                p.*,
                s.name AS supplier_name,
                p.quantity - p.min_quantity AS stock_safety_margin,
                CASE 
                    WHEN p.quantity <= 0 THEN 'Rupture'
                    WHEN p.quantity <= p.min_quantity THEN 'Critique'
                    WHEN p.quantity <= p.reorder_point THEN 'Alerte'
                    WHEN p.quantity >= p.max_quantity THEN 'Surcharge'
                    ELSE 'Normal'
                END AS stock_status,
                p.quantity * p.unit_price AS current_stock_value,
                julianday('now') - julianday(COALESCE(p.last_purchase_date, '2000-01-01')) AS days_since_last_purchase
            FROM spare_parts p
            LEFT JOIN suppliers s ON p.supplier_id = s.id
        """)

        # Vue des fournisseurs avec évaluations
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_suppliers_evaluation AS
            SELECT 
                s.*,
                COUNT(DISTINCT po.id) AS total_orders,
                SUM(po.total_amount) AS total_spent,
                AVG(po.total_amount) AS avg_order_value,
                AVG(julianday(COALESCE(po.actual_delivery_date, 'now')) - julianday(po.order_date)) AS avg_delivery_days,
                COUNT(DISTINCT a.id) AS supplied_assets,
                COUNT(DISTINCT p.id) AS supplied_parts
            FROM suppliers s
            LEFT JOIN purchase_orders po ON s.id = po.supplier_id
            LEFT JOIN assets a ON s.id = a.supplier_id
            LEFT JOIN spare_parts p ON s.id = p.supplier_id
            GROUP BY s.id
        """)

        # Vue des performances des techniciens
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_technician_performance AS
            SELECT 
                u.id,
                u.first_name || ' ' || u.last_name AS technician_name,
                COUNT(DISTINCT i.id) AS total_interventions,
                SUM(CASE WHEN i.status = 'Terminée' THEN 1 ELSE 0 END) AS completed_interventions,
                SUM(CASE WHEN i.status = 'En cours' THEN 1 ELSE 0 END) AS ongoing_interventions,
                AVG(julianday(COALESCE(i.completion_date, 'now')) - julianday(i.opening_date)) AS avg_resolution_days,
                AVG(i.satisfaction_score) AS avg_satisfaction,
                SUM(i.total_cost) AS total_cost,
                SUM(i.actual_duration) AS total_hours,
                SUM(i.parts_cost) AS total_parts_cost,
                SUM(i.labor_cost) AS total_labor_cost,
                COUNT(DISTINCT i.asset_id) AS unique_assets_serviced
            FROM users u
            LEFT JOIN interventions i ON u.id = i.technician_id
            WHERE u.role = 'technician'
            GROUP BY u.id
        """)

        # Vue des coûts par équipement
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_costs_by_asset AS
            SELECT 
                a.id,
                a.code,
                a.name,
                a.purchase_price,
                a.current_value,
                COUNT(DISTINCT i.id) AS intervention_count,
                SUM(i.parts_cost) AS total_parts_cost,
                SUM(i.labor_cost) AS total_labor_cost,
                SUM(i.travel_cost) AS total_travel_cost,
                SUM(i.other_cost) AS total_other_cost,
                SUM(i.total_cost) AS total_maintenance_cost,
                SUM(i.total_cost) / NULLIF(COUNT(DISTINCT i.id), 0) AS avg_cost_per_intervention,
                SUM(i.total_cost) / NULLIF(a.purchase_price, 0) * 100 AS maintenance_cost_ratio
            FROM assets a
            LEFT JOIN interventions i ON a.id = i.asset_id
            GROUP BY a.id
        """)

        # Vue des maintenances préventives
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_preventive_maintenance AS
            SELECT 
                a.id AS asset_id,
                a.code AS asset_code,
                a.name AS asset_name,
                a.next_maintenance_date,
                a.last_maintenance_date,
                a.maintenance_frequency_days,
                julianday(a.next_maintenance_date) - julianday('now') AS days_until_due,
                CASE 
                    WHEN a.next_maintenance_date < date('now') THEN 'En retard'
                    WHEN a.next_maintenance_date <= date('now', '+7 days') THEN 'Imminente'
                    WHEN a.next_maintenance_date <= date('now', '+30 days') THEN 'Proche'
                    ELSE 'Lointaine'
                END AS maintenance_status,
                u.first_name || ' ' || u.last_name AS responsible_name
            FROM assets a
            LEFT JOIN users u ON a.responsible_id = u.id
            WHERE a.next_maintenance_date IS NOT NULL
              AND a.is_active = 1
            ORDER BY a.next_maintenance_date
        """)

        # Vue des mouvements de stock mensuels
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS view_monthly_stock_movements AS
            SELECT 
                strftime('%Y-%m', m.movement_date) AS month,
                p.category,
                SUM(CASE WHEN m.type = 'Entrée' THEN m.quantity ELSE 0 END) AS total_entries,
                SUM(CASE WHEN m.type = 'Sortie' THEN m.quantity ELSE 0 END) AS total_exits,
                SUM(CASE WHEN m.type = 'Entrée' THEN m.total_price ELSE 0 END) AS entry_value,
                SUM(CASE WHEN m.type = 'Sortie' THEN m.total_price ELSE 0 END) AS exit_value,
                COUNT(DISTINCT m.part_id) AS unique_parts_moved,
                COUNT(*) AS total_movements
            FROM stock_movements m
            JOIN spare_parts p ON m.part_id = p.id
            GROUP BY strftime('%Y-%m', m.movement_date), p.category
        """)

    def insert_default_data(self, cursor):
        # Vérifier si des données existent déjà
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            # Créer un administrateur par défaut
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, first_name, last_name, role)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("admin", "admin@gmao.local", password_hash, "Administrateur", "Système", "admin"))

            admin_id = cursor.lastrowid

            # Mettre à jour created_by et updated_by
            cursor.execute("UPDATE users SET created_by = ?, updated_by = ? WHERE id = ?",
                         (admin_id, admin_id, admin_id))

        # Catégories d'équipements par défaut
        cursor.execute("SELECT COUNT(*) FROM asset_categories")
        if cursor.fetchone()[0] == 0:
            categories = [
                ("MACH", "Machines", "Machines de production", None, 1),
                ("EQP", "Équipements", "Équipements divers", None, 1),
                ("VEH", "Véhicules", "Véhicules et engins", None, 1),
                ("OUT", "Outils", "Outils et instruments", None, 1),
                ("INF", "Infrastructure", "Bâtiments et infrastructures", None, 1),
                ("MACH-PROD", "Machines de production", "Machines dédiées à la production", 1, 2),
                ("MACH-EMB", "Machines d'emballage", "Machines d'emballage et conditionnement", 1, 2),
                ("EQP-BUR", "Équipements de bureau", "Mobilier et équipements de bureau", 2, 2),
                ("EQP-INFO", "Équipements informatiques", "Ordinateurs, serveurs, réseau", 2, 2),
                ("VEH-UTIL", "Véhicules utilitaires", "Camions, fourgons", 3, 2),
                ("VEH-ENG", "Engins", "Chariots élévateurs, nacelles", 3, 2),
                ("OUT-MES", "Outils de mesure", "Instruments de mesure et contrôle", 4, 2),
                ("OUT-ELEC", "Outils électriques", "Perceuses, meuleuses", 4, 2),
                ("INF-BAT", "Bâtiments", "Bâtiments administratifs et industriels", 5, 2),
                ("INF-RES", "Réseaux", "Réseaux électriques, fluides", 5, 2)
            ]

            for cat in categories:
                cursor.execute("""
                    INSERT INTO asset_categories (code, name, description, parent_id, level)
                    VALUES (?, ?, ?, ?, ?)
                """, cat)

        # Paramètres par défaut
        cursor.execute("SELECT COUNT(*) FROM settings")
        if cursor.fetchone()[0] == 0:
            settings = [
                ("app_name", "GMAO Enterprise", "string", "general", "Nom de l'application", 1, 0),
                ("app_version", "3.0.0", "string", "general", "Version de l'application", 1, 0),
                ("company_name", "Votre Entreprise", "string", "general", "Nom de l'entreprise", 1, 0),
                ("company_logo", "", "string", "general", "Logo de l'entreprise", 1, 0),
                ("company_address", "", "string", "general", "Adresse de l'entreprise", 1, 0),
                ("company_phone", "", "string", "general", "Téléphone de l'entreprise", 1, 0),
                ("company_email", "", "string", "general", "Email de l'entreprise", 1, 0),
                ("company_siret", "", "string", "general", "SIRET de l'entreprise", 1, 0),
                ("company_vat", "", "string", "general", "N° TVA de l'entreprise", 1, 0),

                ("date_format", "DD/MM/YYYY", "string", "format", "Format des dates", 1, 0),
                ("time_format", "HH:MM", "string", "format", "Format des heures", 1, 0),
                ("datetime_format", "DD/MM/YYYY HH:MM", "string", "format", "Format date/heure", 1, 0),
                ("currency_symbol", "€", "string", "format", "Symbole monétaire", 1, 0),
                ("currency_position", "after", "string", "format", "Position du symbole", 1, 0),
                ("decimal_separator", ",", "string", "format", "Séparateur décimal", 1, 0),
                ("thousand_separator", " ", "string", "format", "Séparateur milliers", 1, 0),
                ("first_day_of_week", "1", "integer", "format", "Premier jour de la semaine", 1, 0),

                ("language", "fr", "string", "localization", "Langue par défaut", 1, 0),
                ("timezone", "Europe/Paris", "string", "localization", "Fuseau horaire", 1, 0),
                ("country", "France", "string", "localization", "Pays", 1, 0),

                ("items_per_page", "20", "integer", "display", "Éléments par page", 1, 0),
                ("default_dashboard", "default", "string", "display", "Dashboard par défaut", 1, 0),
                ("theme", "light", "string", "display", "Thème par défaut", 1, 0),
                ("chart_colors", "#1f77b4,#ff7f0e,#2ca02c,#d62728,#9467bd", "string", "display", "Couleurs des graphiques", 1, 0),

                ("session_timeout", "30", "integer", "security", "Timeout session (minutes)", 1, 0),
                ("password_min_length", "8", "integer", "security", "Longueur min mot de passe", 1, 0),
                ("password_require_uppercase", "1", "boolean", "security", "Exiger majuscule", 1, 0),
                ("password_require_lowercase", "1", "boolean", "security", "Exiger minuscule", 1, 0),
                ("password_require_number", "1", "boolean", "security", "Exiger chiffre", 1, 0),
                ("password_require_special", "1", "boolean", "security", "Exiger caractère spécial", 1, 0),
                ("max_login_attempts", "5", "integer", "security", "Tentatives max connexion", 1, 0),
                ("lockout_duration", "15", "integer", "security", "Durée verrouillage (minutes)", 1, 0),
                ("two_factor_auth", "0", "boolean", "security", "Authentification 2 facteurs", 1, 0),

                ("maintenance_alert_days", "7", "integer", "alerts", "Jours avant maintenance", 1, 0),
                ("stock_alert_threshold", "10", "integer", "alerts", "Seuil alerte stock (%)", 1, 0),
                ("warranty_alert_days", "30", "integer", "alerts", "Jours avant fin garantie", 1, 0),
                ("contract_alert_days", "30", "integer", "alerts", "Jours avant fin contrat", 1, 0),

                ("email_notifications", "1", "boolean", "notifications", "Notifications email", 1, 0),
                ("smtp_server", "smtp.gmail.com", "string", "notifications", "Serveur SMTP", 1, 0),
                ("smtp_port", "587", "integer", "notifications", "Port SMTP", 1, 0),
                ("smtp_username", "", "string", "notifications", "Utilisateur SMTP", 1, 0),
                ("smtp_password", "", "string", "notifications", "Mot de passe SMTP", 1, 0),
                ("smtp_from", "noreply@gmao.local", "string", "notifications", "Email expéditeur", 1, 0),

                ("backup_enabled", "1", "boolean", "backup", "Sauvegarde automatique", 1, 0),
                ("backup_interval", "24", "integer", "backup", "Intervalle sauvegarde (heures)", 1, 0),
                ("backup_retention", "30", "integer", "backup", "Rétention sauvegardes (jours)", 1, 0),
                ("backup_path", "backups/", "string", "backup", "Chemin des sauvegardes", 1, 0),

                ("api_enabled", "1", "boolean", "api", "API activée", 1, 0),
                ("api_rate_limit", "100", "integer", "api", "Limite de requêtes/minute", 1, 0),
                ("api_key_expiry", "365", "integer", "api", "Expiration clés API (jours)", 1, 0),

                ("debug_mode", "0", "boolean", "system", "Mode debug", 1, 0),
                ("log_level", "INFO", "string", "system", "Niveau de log", 1, 0),
                ("log_retention", "30", "integer", "system", "Rétention logs (jours)", 1, 0),
                ("cache_enabled", "1", "boolean", "system", "Cache activé", 1, 0),
                ("cache_ttl", "300", "integer", "system", "Durée cache (secondes)", 1, 0)
            ]

            for setting in settings:
                cursor.execute("""
                    INSERT INTO settings (key, value, type, category, description, is_public, is_system)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, setting)

        # Tâches planifiées par défaut
        cursor.execute("SELECT COUNT(*) FROM scheduled_tasks")
        if cursor.fetchone()[0] == 0:
            tasks = [
                ("Sauvegarde automatique", "backup", "0 2 * * *", '{"type":"full"}', "Sauvegarde quotidienne à 2h"),
                ("Nettoyage des logs", "cleanup", "0 3 * * 0", '{"type":"logs","days":30}', "Nettoyage hebdomadaire des logs"),
                ("Alertes stock", "check_stock", "*/30 * * * *", '{}', "Vérification des alertes stock toutes les 30 minutes"),
                ("Rapport mensuel", "report", "0 8 1 * *", '{"type":"monthly","format":"PDF"}', "Génération rapport mensuel"),
                ("Synchronisation", "sync", "*/15 * * * *", '{"type":"external"}', "Synchronisation externe toutes les 15 minutes")
            ]

            for task in tasks:
                cursor.execute("""
                    INSERT INTO scheduled_tasks (name, task_type, schedule, parameters, description)
                    VALUES (?, ?, ?, ?, ?)
                """, task)

        # Widgets par défaut pour le dashboard
        cursor.execute("SELECT COUNT(*) FROM dashboards")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO dashboards (name, description, layout, is_default)
                VALUES (?, ?, ?, ?)
            """, ("Dashboard par défaut", "Dashboard principal", '{"type":"grid","columns":3}', 1))

            dashboard_id = cursor.lastrowid

            widgets = [
                (dashboard_id, "KPIs", "kpi", 1, '{"width":3}', '{"kpis":["assets","interventions","stock","suppliers"]}'),
                (dashboard_id, "Équipements par statut", "chart", 2, '{"width":1}', '{"type":"pie","data":"assets_by_status"}'),
                (dashboard_id, "Interventions urgentes", "list", 3, '{"width":1}', '{"type":"urgent_interventions","limit":5}'),
                (dashboard_id, "Stock faible", "list", 4, '{"width":1}', '{"type":"low_stock","limit":5}'),
                (dashboard_id, "Maintenances à venir", "calendar", 5, '{"width":2}', '{"days":30}'),
                (dashboard_id, "Coûts par mois", "chart", 6, '{"width":1}', '{"type":"bar","data":"monthly_costs"}')
            ]

            for widget in widgets:
                cursor.execute("""
                    INSERT INTO widgets (dashboard_id, name, type, position, size, configuration)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, widget)

        # Préférences utilisateur par défaut pour l'admin
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        admin = cursor.fetchone()
        if admin:
            admin_id = admin[0]
            cursor.execute("SELECT COUNT(*) FROM user_preferences WHERE user_id = ?", (admin_id,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, theme, language, notifications_enabled, email_notifications, favorites)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (admin_id, "light", "fr", 1, 1, '["dashboard", "assets", "reports"]'))

# =============================================================================
# AUTHENTICATION MANAGER (corrigé)
# =============================================================================

class AuthenticationManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.login_attempts = {}
        self.active_sessions = {}
        self.session_timeout = 3600

    def authenticate(self, username: str, password: str, ip_address: str = None, user_agent: str = None) -> Optional[Dict]:
        try:
            if self.is_locked_out(username):
                logger.warning(f"Compte verrouillé: {username}")
                raise AuthenticationError("Compte temporairement verrouillé. Réessayez plus tard.")

            password_hash = hashlib.sha256(password.encode()).hexdigest()
            query = """
                SELECT id, username, email, first_name, last_name, role, 
                       department, position, phone, mobile, photo,
                       is_active, last_login
                FROM users 
                WHERE username = ? AND password_hash = ? AND is_deleted = 0
            """
            result = self.db.execute_query(query, (username, password_hash))

            if not result.empty:
                user = result.iloc[0].to_dict()
                if not user['is_active']:
                    raise AuthenticationError("Compte désactivé. Contactez l'administrateur.")

                self.db.execute_update(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user['id'],)
                )
                session_id = self.create_session(user['id'], ip_address, user_agent)
                self.log_action(user['id'], 'login', 'users', user['id'],
                                ip_address=ip_address, user_agent=user_agent)
                self.login_attempts.pop(username, None)
                prefs = self.get_user_preferences(user['id'])
                user['preferences'] = prefs
                user['permissions'] = self.get_permissions(user['role'])
                logger.info(f"Connexion réussie: {username}")
                return {'session_id': session_id, 'user': user}
            else:
                self.record_failed_attempt(username)
                logger.warning(f"Tentative de connexion échouée: {username}")
                raise AuthenticationError("Nom d'utilisateur ou mot de passe incorrect")
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Erreur d'authentification: {e}")
            raise AuthenticationError("Erreur lors de l'authentification")

    def create_session(self, user_id: int, ip_address: str = None, user_agent: str = None) -> str:
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
        query = """
            INSERT INTO sessions (session_id, user_id, ip_address, user_agent, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (session_id, user_id, ip_address, user_agent, expires_at))
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'expires_at': expires_at,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict]:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session['expires_at'] > datetime.now():
                return session
        query = """
            SELECT * FROM sessions 
            WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP AND is_active = 1
        """
        result = self.db.execute_query(query, (session_id,))
        if not result.empty:
            session = result.iloc[0].to_dict()
            self.active_sessions[session_id] = {
                'user_id': session['user_id'],
                'expires_at': datetime.fromisoformat(session['expires_at']),
                'ip_address': session['ip_address'],
                'user_agent': session['user_agent']
            }
            return self.active_sessions[session_id]
        return None

    def invalidate_session(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        self.db.execute_update("UPDATE sessions SET is_active = 0 WHERE session_id = ?", (session_id,))

    def record_failed_attempt(self, username: str):
        if username not in self.login_attempts:
            self.login_attempts[username] = {'count': 1, 'first_attempt': datetime.now()}
        else:
            self.login_attempts[username]['count'] += 1

    def is_locked_out(self, username: str) -> bool:
        if username in self.login_attempts:
            attempts = self.login_attempts[username]
            max_attempts = self.get_setting('max_login_attempts', 5)
            lockout_duration = self.get_setting('lockout_duration', 15)
            if attempts['count'] >= max_attempts:
                lockout_time = attempts['first_attempt'] + timedelta(minutes=lockout_duration)
                if datetime.now() < lockout_time:
                    return True
                else:
                    self.login_attempts.pop(username, None)
        return False

    def get_setting(self, key: str, default: Any = None) -> Any:
        query = "SELECT value, type FROM settings WHERE key = ?"
        result = self.db.execute_query(query, (key,))
        if not result.empty:
            value = result.iloc[0]['value']
            type_ = result.iloc[0]['type']
            if type_ == 'integer':
                return int(value)
            elif type_ == 'boolean':
                return value == '1' or value.lower() == 'true'
            elif type_ == 'float':
                return float(value)
            else:
                return value
        return default

    def get_user_preferences(self, user_id: int) -> Dict:
        query = "SELECT * FROM user_preferences WHERE user_id = ?"
        result = self.db.execute_query(query, (user_id,))
        if not result.empty:
            prefs = result.iloc[0].to_dict()
            for field in ['favorites', 'recent_items', 'filters', 'columns_visibility']:
                if prefs.get(field):
                    try:
                        prefs[field] = json.loads(prefs[field])
                    except:
                        prefs[field] = []
            return prefs
        else:
            return self.create_default_preferences(user_id)

    def create_default_preferences(self, user_id: int) -> Dict:
        default_prefs = {
            'user_id': user_id,
            'theme': self.get_setting('theme', 'light'),
            'language': self.get_setting('language', 'fr'),
            'notifications_enabled': True,
            'email_notifications': True,
            'favorites': [],
            'recent_items': [],
            'filters': {},
            'columns_visibility': {}
        }
        query = """
            INSERT INTO user_preferences 
            (user_id, theme, language, notifications_enabled, email_notifications, favorites, recent_items, filters, columns_visibility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (
            user_id,
            default_prefs['theme'],
            default_prefs['language'],
            default_prefs['notifications_enabled'],
            default_prefs['email_notifications'],
            json.dumps(default_prefs['favorites']),
            json.dumps(default_prefs['recent_items']),
            json.dumps(default_prefs['filters']),
            json.dumps(default_prefs['columns_visibility'])
        ))
        return default_prefs

    def save_user_preferences(self, user_id: int, preferences: Dict):
        updates = []
        params = []
        for field in ['theme', 'language', 'notifications_enabled', 'email_notifications']:
            if field in preferences:
                updates.append(f"{field} = ?")
                params.append(preferences[field])
        for field in ['favorites', 'recent_items', 'filters', 'columns_visibility']:
            if field in preferences:
                updates.append(f"{field} = ?")
                params.append(json.dumps(preferences[field]))
        if updates:
            query = f"UPDATE user_preferences SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?"
            params.append(user_id)
            self.db.execute_update(query, tuple(params))

    def get_permissions(self, role: str) -> List[str]:
        permissions = {
            'admin': ['*'],
            'manager': [
                'view_dashboard', 'view_assets', 'create_asset', 'edit_asset', 'delete_asset',
                'view_interventions', 'create_intervention', 'edit_intervention', 'delete_intervention',
                'view_maintenance', 'create_maintenance', 'edit_maintenance', 'delete_maintenance',
                'view_stock', 'create_stock', 'edit_stock', 'delete_stock',
                'view_suppliers', 'create_supplier', 'edit_supplier', 'delete_supplier',
                'view_reports', 'create_report', 'export_data',
                'view_planning', 'edit_planning',
                'view_contracts', 'create_contract', 'edit_contract',
                'view_purchases', 'create_purchase', 'edit_purchase',
                'view_users', 'create_user', 'edit_user',
                'view_documents', 'upload_document', 'delete_document'
            ],
            'supervisor': [
                'view_dashboard', 'view_assets', 'edit_asset',
                'view_interventions', 'create_intervention', 'edit_intervention', 'assign_intervention',
                'view_maintenance', 'create_maintenance', 'edit_maintenance',
                'view_stock', 'edit_stock',
                'view_suppliers', 'edit_supplier',
                'view_reports', 'export_data',
                'view_planning', 'edit_planning',
                'view_contracts',
                'view_purchases',
                'view_documents', 'upload_document'
            ],
            'technician': [
                'view_dashboard', 'view_assets',
                'view_interventions', 'edit_assigned_interventions', 'complete_intervention',
                'view_maintenance', 'complete_maintenance',
                'view_stock', 'use_stock',
                'view_planning',
                'view_documents'
            ],
            'operator': [
                'view_dashboard', 'view_assets',
                'create_intervention', 'view_my_interventions',
                'view_stock',
                'view_planning'
            ],
            'viewer': [
                'view_dashboard', 'view_assets', 'view_interventions',
                'view_maintenance', 'view_stock', 'view_suppliers',
                'view_reports', 'view_planning', 'view_documents'
            ],
            'auditor': [
                'view_dashboard', 'view_assets', 'view_interventions',
                'view_maintenance', 'view_stock', 'view_suppliers',
                'view_reports', 'view_contracts', 'view_purchases',
                'view_audit_logs', 'export_data'
            ],
            'accountant': [
                'view_dashboard', 'view_assets',
                'view_interventions', 'view_costs',
                'view_stock', 'view_value',
                'view_suppliers', 'view_prices',
                'view_reports', 'view_financial',
                'view_contracts', 'view_purchases',
                'export_data'
            ],
            'purchaser': [
                'view_dashboard', 'view_assets',
                'view_stock', 'edit_stock',
                'view_suppliers', 'edit_supplier',
                'view_reports', 'view_purchases', 'create_purchase', 'edit_purchase',
                'view_contracts', 'create_contract', 'edit_contract'
            ],
            'stock_manager': [
                'view_dashboard',
                'view_stock', 'create_stock', 'edit_stock', 'delete_stock',
                'view_movements', 'create_movement',
                'view_inventory', 'create_inventory',
                'view_reports', 'export_data',
                'view_suppliers'
            ]
        }
        return permissions.get(role, [])

    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        if '*' in user_permissions:
            return True
        return required_permission in user_permissions

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        try:
            old_hash = hashlib.sha256(old_password.encode()).hexdigest()
            query = "SELECT id FROM users WHERE id = ? AND password_hash = ?"
            result = self.db.execute_query(query, (user_id, old_hash))
            if result.empty:
                return False, "Ancien mot de passe incorrect"
            is_valid, message = self.validate_password(new_password)
            if not is_valid:
                return False, message
            new_hash = hashlib.sha256(new_password.encode()).hexdigest()
            self.db.execute_update(
                "UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_hash, user_id)
            )
            self.log_action(user_id, 'password_change', 'users', user_id)
            logger.info(f"Changement de mot de passe pour l'utilisateur {user_id}")
            return True, "Mot de passe changé avec succès"
        except Exception as e:
            logger.error(f"Erreur changement mot de passe: {e}")
            return False, f"Erreur: {str(e)}"

    def validate_password(self, password: str) -> Tuple[bool, str]:
        min_length = self.get_setting('password_min_length', 8)
        require_uppercase = self.get_setting('password_require_uppercase', True)
        require_lowercase = self.get_setting('password_require_lowercase', True)
        require_number = self.get_setting('password_require_number', True)
        require_special = self.get_setting('password_require_special', True)
        if len(password) < min_length:
            return False, f"Le mot de passe doit contenir au moins {min_length} caractères"
        if require_uppercase and not re.search(r"[A-Z]", password):
            return False, "Le mot de passe doit contenir au moins une majuscule"
        if require_lowercase and not re.search(r"[a-z]", password):
            return False, "Le mot de passe doit contenir au moins une minuscule"
        if require_number and not re.search(r"[0-9]", password):
            return False, "Le mot de passe doit contenir au moins un chiffre"
        if require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Le mot de passe doit contenir au moins un caractère spécial"
        return True, "Mot de passe valide"

    def reset_password(self, email: str) -> Tuple[bool, str]:
        try:
            query = "SELECT id, username, first_name FROM users WHERE email = ? AND is_deleted = 0"
            result = self.db.execute_query(query, (email,))
            if result.empty:
                return False, "Email non trouvé"
            user = result.iloc[0].to_dict()
            token = secrets.token_urlsafe(32)
            expiry = datetime.now() + timedelta(hours=24)
            # TODO: Sauvegarder le token et envoyer l'email
            logger.info(f"Demande de réinitialisation pour {email}")
            return True, "Email de réinitialisation envoyé"
        except Exception as e:
            logger.error(f"Erreur réinitialisation mot de passe: {e}")
            return False, f"Erreur: {str(e)}"

    def create_user(self, user_data: Dict) -> Tuple[bool, Union[int, str]]:
        try:
            checks = [
                ("username", user_data.get('username')),
                ("email", user_data.get('email'))
            ]
            for field, value in checks:
                if value:
                    query = f"SELECT id FROM users WHERE {field} = ? AND is_deleted = 0"
                    result = self.db.execute_query(query, (value,))
                    if not result.empty:
                        return False, f"{field} déjà utilisé"
            if 'password' in user_data and user_data['password']:
                is_valid, message = self.validate_password(user_data['password'])
                if not is_valid:
                    return False, message
                password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
            else:
                temp_password = secrets.token_urlsafe(12)
                password_hash = hashlib.sha256(temp_password.encode()).hexdigest()
                user_data['temp_password'] = temp_password
            query = """
                INSERT INTO users (
                    username, email, password_hash, first_name, last_name, role,
                    department, position, phone, mobile, address, city, postal_code, country,
                    hire_date, birth_date, emergency_contact, emergency_phone,
                    notes, is_active, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                user_data.get('username'),
                user_data.get('email'),
                password_hash,
                user_data.get('first_name'),
                user_data.get('last_name'),
                user_data.get('role', 'viewer'),
                user_data.get('department'),
                user_data.get('position'),
                user_data.get('phone'),
                user_data.get('mobile'),
                user_data.get('address'),
                user_data.get('city'),
                user_data.get('postal_code'),
                user_data.get('country', 'France'),
                user_data.get('hire_date'),
                user_data.get('birth_date'),
                user_data.get('emergency_contact'),
                user_data.get('emergency_phone'),
                user_data.get('notes'),
                user_data.get('is_active', 1),
                user_data.get('created_by')
            )
            user_id = self.db.execute_insert(query, params)
            self.create_default_preferences(user_id)
            self.log_action(user_data.get('created_by'), 'create', 'users', user_id,
                            new_values=json.dumps(user_data))
            logger.info(f"Utilisateur créé: {user_data.get('username')}")
            if 'temp_password' in user_data:
                return True, f"Utilisateur créé avec mot de passe temporaire: {user_data['temp_password']}"
            else:
                return True, user_id
        except Exception as e:
            logger.error(f"Erreur création utilisateur: {e}")
            return False, str(e)

    def update_user(self, user_id: int, user_data: Dict, updater_id: int = None) -> Tuple[bool, str]:
        try:
            current = self.get_user_by_id(user_id)
            if not current:
                return False, "Utilisateur non trouvé"
            updates = []
            params = []
            updateable_fields = [
                'email', 'first_name', 'last_name', 'role', 'department', 'position',
                'phone', 'mobile', 'address', 'city', 'postal_code', 'country',
                'hire_date', 'birth_date', 'emergency_contact', 'emergency_phone',
                'photo', 'signature', 'notes', 'is_active'
            ]
            for field in updateable_fields:
                if field in user_data:
                    updates.append(f"{field} = ?")
                    params.append(user_data[field])
            if 'password' in user_data and user_data['password']:
                is_valid, message = self.validate_password(user_data['password'])
                if not is_valid:
                    return False, message
                password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
                updates.append("password_hash = ?")
                params.append(password_hash)
            if not updates:
                return False, "Aucune donnée à mettre à jour"
            query = f"UPDATE users SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?"
            params.append(updater_id)
            params.append(user_id)
            self.db.execute_update(query, tuple(params))
            changes = {}
            for field in user_data:
                if field in current and str(current[field]) != str(user_data[field]):
                    changes[field] = {'old': current[field], 'new': user_data[field]}
            if changes:
                self.log_action(updater_id, 'update', 'users', user_id,
                              old_values=json.dumps({k: v['old'] for k, v in changes.items()}),
                              new_values=json.dumps({k: v['new'] for k, v in changes.items()}))
            logger.info(f"Utilisateur mis à jour: {user_id}")
            return True, "Utilisateur mis à jour"
        except Exception as e:
            logger.error(f"Erreur mise à jour utilisateur: {e}")
            return False, str(e)

    def delete_user(self, user_id: int, deleter_id: int = None) -> bool:
        try:
            result = self.db.execute_update(
                "UPDATE users SET is_active = 0, is_deleted = 1, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?",
                (deleter_id, user_id)
            )
            if result > 0:
                self.db.execute_update("UPDATE sessions SET is_active = 0 WHERE user_id = ?", (user_id,))
                self.log_action(deleter_id, 'delete', 'users', user_id)
                logger.info(f"Utilisateur désactivé: {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur suppression utilisateur: {e}")
            return False

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        query = """
            SELECT id, username, email, first_name, last_name, role,
                   department, position, phone, mobile, address, city,
                   postal_code, country, hire_date, birth_date,
                   emergency_contact, emergency_phone, photo, signature,
                   notes, is_active, last_login, created_at, updated_at,
                   created_by, updated_by
            FROM users 
            WHERE id = ? AND is_deleted = 0
        """
        result = self.db.execute_query(query, (user_id,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        query = """
            SELECT id, username, email, first_name, last_name, role,
                   department, position, phone, mobile, is_active
            FROM users 
            WHERE username = ? AND is_deleted = 0
        """
        result = self.db.execute_query(query, (username,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_all_users(self, active_only: bool = True) -> pd.DataFrame:
        query = """
            SELECT id, username, email, first_name || ' ' || last_name as full_name,
                   first_name, last_name, role, department, position,
                   phone, mobile, is_active, last_login, created_at
            FROM users
            WHERE is_deleted = 0
        """
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY last_name, first_name"
        return self.db.execute_query(query)

    def log_action(self, user_id: int, action: str, entity_type: str = None,
                  entity_id: int = None, old_values: str = None,
                  new_values: str = None, ip_address: str = None,
                  user_agent: str = None):
        query = """
            INSERT INTO audits (user_id, action, entity_type, entity_id,
                              old_value, new_value, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (user_id, action, entity_type, entity_id,
                                      old_values, new_values, ip_address, user_agent))

# =============================================================================
# ASSET MANAGER (corrigé)
# =============================================================================

class AssetManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_asset(self, asset_data: Dict, user_id: int = None) -> Tuple[bool, Union[int, str]]:
        try:
            if 'code' not in asset_data or not asset_data['code']:
                asset_data['code'] = self.generate_asset_code(asset_data.get('type', 'EQ'))
            if 'barcode' not in asset_data or not asset_data['barcode']:
                asset_data['barcode'] = self.generate_barcode()
            if 'qr_code' not in asset_data or not asset_data['qr_code']:
                asset_data['qr_code'] = self.generate_qr_code(asset_data['code'])

            if 'purchase_price' in asset_data and 'depreciation_rate' in asset_data:
                asset_data['current_value'] = self.calculate_current_value(
                    asset_data['purchase_price'],
                    asset_data.get('acquisition_date'),
                    asset_data['depreciation_rate']
                )

            if asset_data.get('commissioning_date') and asset_data.get('maintenance_frequency_days'):
                try:
                    commissioning = datetime.strptime(asset_data['commissioning_date'], '%Y-%m-%d').date()
                    next_date = commissioning + timedelta(days=asset_data['maintenance_frequency_days'])
                    asset_data['next_maintenance_date'] = next_date.isoformat()
                except:
                    pass

            is_valid, message = self.validate_asset_data(asset_data)
            if not is_valid:
                return False, message

            query = """
                INSERT INTO assets (
                    code, name, type, model, manufacturer, serial_number,
                    barcode, qr_code, rfid_tag, acquisition_date, commissioning_date,
                    warranty_end_date, warranty_days, location, department,
                    building, floor, room, responsible_id, status,
                    purchase_price, current_value, depreciation_rate, useful_life_years,
                    last_maintenance_date, next_maintenance_date, maintenance_frequency_days,
                    meter_type, current_meter_value, meter_unit, meter_reset_date,
                    supplier_id, manufacturer_id, category_id, subcategory_id,
                    criticality, energy_consumption, co2_emission,
                    documentation, photo, technical_sheet, notes,
                    is_active, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                asset_data.get('code'),
                asset_data.get('name'),
                asset_data.get('type'),
                asset_data.get('model'),
                asset_data.get('manufacturer'),
                asset_data.get('serial_number'),
                asset_data.get('barcode'),
                asset_data.get('qr_code'),
                asset_data.get('rfid_tag'),
                asset_data.get('acquisition_date'),
                asset_data.get('commissioning_date'),
                asset_data.get('warranty_end_date'),
                asset_data.get('warranty_days', 0),
                asset_data.get('location'),
                asset_data.get('department'),
                asset_data.get('building'),
                asset_data.get('floor'),
                asset_data.get('room'),
                asset_data.get('responsible_id'),
                asset_data.get('status', 'Actif'),
                asset_data.get('purchase_price', 0),
                asset_data.get('current_value', 0),
                asset_data.get('depreciation_rate', 0),
                asset_data.get('useful_life_years', 0),
                asset_data.get('last_maintenance_date'),
                asset_data.get('next_maintenance_date'),
                asset_data.get('maintenance_frequency_days', 0),
                asset_data.get('meter_type'),
                asset_data.get('current_meter_value', 0),
                asset_data.get('meter_unit'),
                asset_data.get('meter_reset_date'),
                asset_data.get('supplier_id'),
                asset_data.get('manufacturer_id'),
                asset_data.get('category_id'),
                asset_data.get('subcategory_id'),
                asset_data.get('criticality', 'Normal'),
                asset_data.get('energy_consumption', 0),
                asset_data.get('co2_emission', 0),
                asset_data.get('documentation'),
                asset_data.get('photo'),
                asset_data.get('technical_sheet'),
                asset_data.get('notes'),
                1,  # is_active
                user_id
            )
            asset_id = self.db.execute_insert(query, params)
            self.log_action(user_id, 'create', 'asset', asset_id, new_values=json.dumps(asset_data))
            logger.info(f"Équipement créé: {asset_data['code']}")
            return True, asset_id
        except Exception as e:
            logger.error(f"Erreur création équipement: {e}")
            return False, str(e)

    def update_asset(self, asset_id: int, asset_data: Dict, user_id: int = None) -> Tuple[bool, str]:
        try:
            current = self.get_asset_by_id(asset_id)
            if not current:
                return False, "Équipement non trouvé"

            if 'purchase_price' in asset_data or 'depreciation_rate' in asset_data:
                purchase_price = asset_data.get('purchase_price', current['purchase_price'])
                depreciation_rate = asset_data.get('depreciation_rate', current['depreciation_rate'])
                acquisition_date = asset_data.get('acquisition_date', current['acquisition_date'])
                asset_data['current_value'] = self.calculate_current_value(
                    purchase_price, acquisition_date, depreciation_rate
                )

            if 'last_maintenance_date' in asset_data or 'maintenance_frequency_days' in asset_data:
                last_date = asset_data.get('last_maintenance_date', current['last_maintenance_date'])
                frequency = asset_data.get('maintenance_frequency_days', current['maintenance_frequency_days'])
                if last_date and frequency:
                    try:
                        last = datetime.strptime(last_date, '%Y-%m-%d').date()
                        next_date = last + timedelta(days=frequency)
                        asset_data['next_maintenance_date'] = next_date.isoformat()
                    except:
                        pass

            updates = []
            params = []
            updateable_fields = [
                'name', 'type', 'model', 'manufacturer', 'serial_number',
                'barcode', 'qr_code', 'rfid_tag', 'acquisition_date',
                'commissioning_date', 'warranty_end_date', 'warranty_days',
                'location', 'department', 'building', 'floor', 'room',
                'responsible_id', 'status', 'purchase_price', 'current_value',
                'depreciation_rate', 'useful_life_years', 'last_maintenance_date',
                'next_maintenance_date', 'maintenance_frequency_days',
                'meter_type', 'current_meter_value', 'meter_unit', 'meter_reset_date',
                'supplier_id', 'manufacturer_id', 'category_id', 'subcategory_id',
                'criticality', 'energy_consumption', 'co2_emission',
                'documentation', 'photo', 'technical_sheet', 'notes', 'is_active'
            ]
            for field in updateable_fields:
                if field in asset_data:
                    updates.append(f"{field} = ?")
                    params.append(asset_data[field])

            if not updates:
                return False, "Aucune donnée à mettre à jour"

            query = f"UPDATE assets SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?"
            params.append(user_id)
            params.append(asset_id)
            self.db.execute_update(query, tuple(params))

            changes = {}
            for field in asset_data:
                if field in current and str(current[field]) != str(asset_data[field]):
                    changes[field] = {'old': current[field], 'new': asset_data[field]}

            if changes:
                self.log_action(user_id, 'update', 'asset', asset_id,
                              old_values=json.dumps({k: v['old'] for k, v in changes.items()}),
                              new_values=json.dumps({k: v['new'] for k, v in changes.items()}))

            logger.info(f"Équipement mis à jour: {asset_id}")
            return True, "Équipement mis à jour"
        except Exception as e:
            logger.error(f"Erreur mise à jour équipement: {e}")
            return False, str(e)

    def delete_asset(self, asset_id: int, user_id: int = None) -> Tuple[bool, str]:
        try:
            has_deps = self.check_dependencies(asset_id)
            if has_deps:
                return False, "Impossible de supprimer: l'équipement a des interventions ou maintenances associées"

            result = self.db.execute_update(
                "UPDATE assets SET is_active = 0, is_deleted = 1, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?",
                (user_id, asset_id)
            )
            if result > 0:
                self.log_action(user_id, 'delete', 'asset', asset_id)
                logger.info(f"Équipement désactivé: {asset_id}")
                return True, "Équipement supprimé"
            return False, "Équipement non trouvé"
        except Exception as e:
            logger.error(f"Erreur suppression équipement: {e}")
            return False, str(e)

    def get_asset_by_id(self, asset_id: int) -> Optional[Dict]:
        query = """
            SELECT a.*, 
                   u.first_name || ' ' || u.last_name as responsible_name,
                   s.name as supplier_name,
                   m.name as manufacturer_name,
                   c.name as category_name,
                   sc.name as subcategory_name
            FROM assets a
            LEFT JOIN users u ON a.responsible_id = u.id
            LEFT JOIN suppliers s ON a.supplier_id = s.id
            LEFT JOIN suppliers m ON a.manufacturer_id = m.id
            LEFT JOIN asset_categories c ON a.category_id = c.id
            LEFT JOIN asset_categories sc ON a.subcategory_id = sc.id
            WHERE a.id = ? AND a.is_deleted = 0
        """
        result = self.db.execute_query(query, (asset_id,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_asset_by_code(self, code: str) -> Optional[Dict]:
        query = "SELECT * FROM assets WHERE code = ? AND is_deleted = 0"
        result = self.db.execute_query(query, (code,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_asset_by_barcode(self, barcode: str) -> Optional[Dict]:
        query = "SELECT * FROM assets WHERE barcode = ? AND is_deleted = 0"
        result = self.db.execute_query(query, (barcode,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_all_assets(self, filters: Dict = None) -> pd.DataFrame:
        query = """
            SELECT a.*, 
                   u.first_name || ' ' || u.last_name as responsible_name,
                   s.name as supplier_name,
                   c.name as category_name
            FROM assets a
            LEFT JOIN users u ON a.responsible_id = u.id
            LEFT JOIN suppliers s ON a.supplier_id = s.id
            LEFT JOIN asset_categories c ON a.category_id = c.id
            WHERE a.is_deleted = 0
        """
        params = []
        if filters:
            if filters.get('status'):
                query += " AND a.status = ?"
                params.append(filters['status'])
            if filters.get('type'):
                query += " AND a.type = ?"
                params.append(filters['type'])
            if filters.get('category_id'):
                query += " AND a.category_id = ?"
                params.append(filters['category_id'])
            if filters.get('responsible_id'):
                query += " AND a.responsible_id = ?"
                params.append(filters['responsible_id'])
            if filters.get('supplier_id'):
                query += " AND a.supplier_id = ?"
                params.append(filters['supplier_id'])
            if filters.get('location'):
                query += " AND a.location LIKE ?"
                params.append(f'%{filters["location"]}%')
            if filters.get('department'):
                query += " AND a.department = ?"
                params.append(filters['department'])
            if filters.get('is_active') is not None:
                query += " AND a.is_active = ?"
                params.append(1 if filters['is_active'] else 0)
            if filters.get('maintenance_due'):
                query += " AND a.next_maintenance_date <= date('now', '+7 days')"
            if filters.get('warranty_expiring'):
                query += " AND a.warranty_end_date <= date('now', '+30 days')"
            if filters.get('search'):
                search = f"%{filters['search']}%"
                query += """ AND (a.name LIKE ? OR a.code LIKE ? OR 
                              a.model LIKE ? OR a.serial_number LIKE ? OR
                              a.manufacturer LIKE ?)"""
                params.extend([search, search, search, search, search])
        query += " ORDER BY a.name"
        return self.db.execute_query(query, tuple(params) if params else None)

    def get_assets_due_for_maintenance(self, days: int = 7) -> pd.DataFrame:
        query = """
            SELECT a.*, 
                   u.first_name || ' ' || u.last_name as responsible_name,
                   julianday(a.next_maintenance_date) - julianday('now') as days_until_due
            FROM assets a
            LEFT JOIN users u ON a.responsible_id = u.id
            WHERE a.is_active = 1
              AND a.is_deleted = 0
              AND a.next_maintenance_date IS NOT NULL
              AND a.next_maintenance_date <= date('now', ?)
            ORDER BY a.next_maintenance_date
        """
        return self.db.execute_query(query, (f'+{days} days',))

    def get_assets_by_status(self) -> pd.DataFrame:
        query = """
            SELECT status, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM assets
            WHERE is_deleted = 0
            GROUP BY status
            ORDER BY count DESC
        """
        return self.db.execute_query(query)

    def get_assets_by_type(self) -> pd.DataFrame:
        query = """
            SELECT type, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM assets
            WHERE is_deleted = 0
            GROUP BY type
            ORDER BY count DESC
        """
        return self.db.execute_query(query)

    def get_assets_by_category(self) -> pd.DataFrame:
        query = """
            SELECT c.name as category, COUNT(a.id) as count,
                   ROUND(COUNT(a.id) * 100.0 / SUM(COUNT(a.id)) OVER(), 1) as percentage
            FROM asset_categories c
            LEFT JOIN assets a ON c.id = a.category_id AND a.is_deleted = 0
            WHERE c.level = 1
            GROUP BY c.id, c.name
            ORDER BY count DESC
        """
        return self.db.execute_query(query)

    def get_assets_by_department(self) -> pd.DataFrame:
        query = """
            SELECT COALESCE(department, 'Non assigné') as department,
                   COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM assets
            WHERE is_deleted = 0
            GROUP BY department
            ORDER BY count DESC
        """
        return self.db.execute_query(query)

    def get_assets_by_location(self) -> pd.DataFrame:
        query = """
            SELECT COALESCE(location, 'Non localisé') as location,
                   COUNT(*) as count
            FROM assets
            WHERE is_deleted = 0
            GROUP BY location
            ORDER BY count DESC
            LIMIT 10
        """
        return self.db.execute_query(query)

    def get_assets_by_responsible(self) -> pd.DataFrame:
        query = """
            SELECT u.first_name || ' ' || u.last_name as responsible,
                   COUNT(a.id) as count
            FROM users u
            LEFT JOIN assets a ON u.id = a.responsible_id AND a.is_deleted = 0
            WHERE u.role IN ('manager', 'supervisor', 'technician')
            GROUP BY u.id, responsible
            HAVING count > 0
            ORDER BY count DESC
        """
        return self.db.execute_query(query)

    def get_maintenance_history(self, asset_id: int) -> pd.DataFrame:
        query = """
            SELECT i.id, i.number, i.title, i.type, i.status,
                   i.opening_date, i.completion_date,
                   i.actual_duration, i.total_cost,
                   u.first_name || ' ' || u.last_name as technician_name
            FROM interventions i
            LEFT JOIN users u ON i.technician_id = u.id
            WHERE i.asset_id = ?
              AND i.is_preventive = 1
            ORDER BY i.opening_date DESC
        """
        return self.db.execute_query(query, (asset_id,))

    def get_intervention_history(self, asset_id: int) -> pd.DataFrame:
        query = """
            SELECT i.id, i.number, i.title, i.type, i.priority, i.status,
                   i.opening_date, i.completion_date,
                   i.actual_duration, i.total_cost,
                   u.first_name || ' ' || u.last_name as technician_name,
                   i.satisfaction_score
            FROM interventions i
            LEFT JOIN users u ON i.technician_id = u.id
            WHERE i.asset_id = ?
            ORDER BY i.opening_date DESC
        """
        return self.db.execute_query(query, (asset_id,))

    def get_meter_readings(self, asset_id: int, limit: int = 10) -> pd.DataFrame:
        query = """
            SELECT * FROM meter_readings
            WHERE asset_id = ?
            ORDER BY reading_date DESC
            LIMIT ?
        """
        return self.db.execute_query(query, (asset_id, limit))

    def add_meter_reading(self, asset_id: int, reading_data: Dict, user_id: int = None) -> Tuple[bool, Union[int, str]]:
        try:
            last = self.db.execute_query("""
                SELECT current_value FROM meter_readings
                WHERE asset_id = ?
                ORDER BY reading_date DESC
                LIMIT 1
            """, (asset_id,))
            previous_value = last.iloc[0]['current_value'] if not last.empty else reading_data.get('initial_value', 0)
            current_value = reading_data['current_value']
            difference = current_value - previous_value

            query = """
                INSERT INTO meter_readings (
                    asset_id, meter_type, previous_value, current_value,
                    difference, reading_date, reading_method, reader_id,
                    notes, is_verified, verified_by, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                asset_id,
                reading_data.get('meter_type'),
                previous_value,
                current_value,
                difference,
                reading_data.get('reading_date', datetime.now()),
                reading_data.get('reading_method', 'Manuel'),
                user_id,
                reading_data.get('notes'),
                reading_data.get('is_verified', False),
                reading_data.get('verified_by'),
                reading_data.get('verified_at')
            )
            reading_id = self.db.execute_insert(query, params)
            self.db.execute_update(
                "UPDATE assets SET current_meter_value = ? WHERE id = ?",
                (current_value, asset_id)
            )
            logger.info(f"Relevé de compteur ajouté: {reading_id}")
            return True, reading_id
        except Exception as e:
            logger.error(f"Erreur ajout relevé compteur: {e}")
            return False, str(e)

    def calculate_current_value(self, purchase_price: float, acquisition_date: str, depreciation_rate: float) -> float:
        if not acquisition_date or depreciation_rate <= 0:
            return purchase_price
        try:
            acq_date = datetime.strptime(acquisition_date, '%Y-%m-%d').date()
            days_owned = (date.today() - acq_date).days
            years_owned = days_owned / 365.25
            depreciation = purchase_price * (depreciation_rate / 100) * years_owned
            current_value = max(purchase_price - depreciation, 0)
            return round(current_value, 2)
        except:
            return purchase_price

    def generate_asset_code(self, asset_type: str) -> str:
        prefix = asset_type[:3].upper() if asset_type else 'AST'
        query = """
            SELECT code FROM assets 
            WHERE code LIKE ? 
            ORDER BY code DESC 
            LIMIT 1
        """
        result = self.db.execute_query(query, (f"{prefix}%",))
        if not result.empty:
            last_code = result.iloc[0]['code']
            try:
                last_num = int(last_code[len(prefix):])
                new_num = last_num + 1
            except:
                new_num = 1
        else:
            new_num = 1
        return f"{prefix}{new_num:05d}"

    def generate_barcode(self) -> str:
        import random
        while True:
            code = ''.join([str(random.randint(0, 9)) for _ in range(12)])
            total = 0
            for i, digit in enumerate(code):
                if i % 2 == 0:
                    total += int(digit)
                else:
                    total += int(digit) * 3
            check_digit = (10 - (total % 10)) % 10
            barcode = code + str(check_digit)
            existing = self.db.execute_query(
                "SELECT id FROM assets WHERE barcode = ? UNION SELECT id FROM spare_parts WHERE barcode = ?",
                (barcode, barcode)
            )
            if existing.empty:
                return barcode

def generate_qr_code(self, data: str) -> str:
    """Génère un QR code (version simplifiée sans qrcode)"""
    # Version simplifiée qui retourne juste une chaîne
    return f"QR_CODE_DATA_{data}"
    
    # Version originale (à décommenter quand qrcode sera installé)
    """
    qr_dir = Path("static/qrcodes")
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    filename = f"qr_{uuid.uuid4().hex[:8]}.png"
    filepath = qr_dir / filename
    img.save(filepath)
    return str(filepath)
    """

    def validate_asset_data(self, data: Dict) -> Tuple[bool, str]:
        required_fields = ['name', 'type']
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f"Le champ {field} est requis"
        date_fields = ['acquisition_date', 'commissioning_date', 'warranty_end_date']
        for field in date_fields:
            if field in data and data[field]:
                try:
                    datetime.strptime(data[field], '%Y-%m-%d')
                except:
                    return False, f"Format de date invalide pour {field}"
        numeric_fields = ['purchase_price', 'depreciation_rate', 'useful_life_years',
                         'maintenance_frequency_days', 'energy_consumption']
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    float(data[field])
                except:
                    return False, f"{field} doit être un nombre"
        if data.get('responsible_id'):
            user = self.db.execute_query(
                "SELECT id FROM users WHERE id = ? AND is_active = 1",
                (data['responsible_id'],)
            )
            if user.empty:
                return False, "Responsable invalide"
        if data.get('supplier_id'):
            supplier = self.db.execute_query(
                "SELECT id FROM suppliers WHERE id = ? AND is_active = 1",
                (data['supplier_id'],)
            )
            if supplier.empty:
                return False, "Fournisseur invalide"
        return True, "Données valides"

    def check_dependencies(self, asset_id: int) -> bool:
        interventions = self.db.execute_query(
            "SELECT COUNT(*) as count FROM interventions WHERE asset_id = ?",
            (asset_id,)
        )
        if not interventions.empty and interventions.iloc[0]['count'] > 0:
            return True
        maintenances = self.db.execute_query(
            "SELECT COUNT(*) as count FROM scheduled_tasks WHERE parameters LIKE ?",
            (f'%{asset_id}%',)
        )
        if not maintenances.empty and maintenances.iloc[0]['count'] > 0:
            return True
        return False

    def get_asset_stats(self) -> Dict:
        stats = {}
        total = self.db.execute_query("SELECT COUNT(*) as total FROM assets WHERE is_deleted = 0")
        stats['total'] = int(total['total'].iloc[0]) if not total.empty else 0
        actifs = self.db.execute_query("""
            SELECT COUNT(*) as count FROM assets 
            WHERE status = 'Actif' AND is_deleted = 0
        """)
        stats['actifs'] = int(actifs['count'].iloc[0]) if not actifs.empty else 0
        maintenance = self.db.execute_query("""
            SELECT COUNT(*) as count FROM assets 
            WHERE status = 'En maintenance' AND is_deleted = 0
        """)
        stats['en_maintenance'] = int(maintenance['count'].iloc[0]) if not maintenance.empty else 0
        hs = self.db.execute_query("""
            SELECT COUNT(*) as count FROM assets 
            WHERE status = 'Hors service' AND is_deleted = 0
        """)
        stats['hors_service'] = int(hs['count'].iloc[0]) if not hs.empty else 0
        due = self.get_assets_due_for_maintenance()
        stats['maintenance_due'] = len(due)
        valeur = self.db.execute_query("""
            SELECT SUM(current_value) as total FROM assets 
            WHERE is_deleted = 0
        """)
        stats['valeur_totale'] = float(valeur['total'].iloc[0]) if not valeur.empty and valeur['total'].iloc[0] else 0
        age = self.db.execute_query("""
            SELECT AVG(julianday('now') - julianday(commissioning_date)) as avg_age
            FROM assets
            WHERE commissioning_date IS NOT NULL AND is_deleted = 0
        """)
        stats['age_moyen'] = float(age['avg_age'].iloc[0]) / 365.25 if not age.empty and age['avg_age'].iloc[0] else 0
        return stats

    def log_action(self, user_id: int, action: str, entity_type: str, entity_id: int,
                  old_values: str = None, new_values: str = None):
        query = """
            INSERT INTO histories (entity_type, entity_id, action, user_id, old_values, new_values)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (entity_type, entity_id, action, user_id, old_values, new_values))

    def export_assets(self, format: str = 'csv', filters: Dict = None) -> Union[str, bytes]:
        assets = self.get_all_assets(filters)
        if format == 'csv':
            return assets.to_csv(index=False, encoding='utf-8-sig')
        elif format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                assets.to_excel(writer, sheet_name='Équipements', index=False)
                worksheet = writer.sheets['Équipements']
                for i, col in enumerate(assets.columns):
                    max_len = max(assets[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, min(max_len, 50))
            return output.getvalue()
        elif format == 'json':
            return assets.to_json(orient='records', indent=2, force_ascii=False)
        elif format == 'html':
            return assets.to_html(escape=False, index=False)
        elif format == 'markdown':
            return assets.to_markdown(index=False)
        else:
            return None

# =============================================================================
# INTERVENTION MANAGER (corrigé)
# =============================================================================

class InterventionManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_intervention(self, intervention_data: Dict, user_id: int = None) -> Tuple[bool, Union[int, str]]:
        try:
            if 'number' not in intervention_data or not intervention_data['number']:
                intervention_data['number'] = self.generate_intervention_number()

            is_valid, message = self.validate_intervention_data(intervention_data)
            if not is_valid:
                return False, message

            total = (intervention_data.get('parts_cost', 0) +
                    intervention_data.get('labor_cost', 0) +
                    intervention_data.get('travel_cost', 0) +
                    intervention_data.get('other_cost', 0))
            intervention_data['total_cost'] = total

            query = """
                INSERT INTO interventions (
                    number, title, description, type, priority, status,
                    asset_id, requester_id, technician_id, supervisor_id,
                    opening_date, due_date, estimated_duration,
                    cause, solution, observations, work_performed,
                    parts_used, parts_cost, labor_cost, travel_cost,
                    other_cost, total_cost, requires_followup,
                    is_urgent, is_planned, is_preventive, is_corrective,
                    is_warranty, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                intervention_data.get('number'),
                intervention_data.get('title'),
                intervention_data.get('description'),
                intervention_data.get('type'),
                intervention_data.get('priority', 'Normale'),
                intervention_data.get('status', 'Ouverte'),
                intervention_data.get('asset_id'),
                intervention_data.get('requester_id'),
                intervention_data.get('technician_id'),
                intervention_data.get('supervisor_id'),
                intervention_data.get('opening_date', datetime.now()),
                intervention_data.get('due_date'),
                intervention_data.get('estimated_duration', 0),
                intervention_data.get('cause'),
                intervention_data.get('solution'),
                intervention_data.get('observations'),
                intervention_data.get('work_performed'),
                intervention_data.get('parts_used'),
                intervention_data.get('parts_cost', 0),
                intervention_data.get('labor_cost', 0),
                intervention_data.get('travel_cost', 0),
                intervention_data.get('other_cost', 0),
                total,
                intervention_data.get('requires_followup', False),
                intervention_data.get('is_urgent', False),
                intervention_data.get('is_planned', False),
                intervention_data.get('is_preventive', False),
                intervention_data.get('is_corrective', False),
                intervention_data.get('is_warranty', False),
                user_id
            )
            intervention_id = self.db.execute_insert(query, params)

            if intervention_data.get('is_preventive') and intervention_data.get('completion_date'):
                self.update_asset_maintenance_date(
                    intervention_data['asset_id'],
                    intervention_data.get('completion_date')
                )

            self.log_action(user_id, 'create', 'intervention', intervention_id,
                          new_values=json.dumps(intervention_data))

            if intervention_data.get('technician_id'):
                self.create_notification(
                    intervention_data['technician_id'],
                    'Nouvelle intervention',
                    f"Vous avez été assigné à l'intervention: {intervention_data['title']}",
                    f"/interventions/{intervention_id}"
                )

            logger.info(f"Intervention créée: {intervention_data['number']}")
            return True, intervention_id
        except Exception as e:
            logger.error(f"Erreur création intervention: {e}")
            return False, str(e)

    def update_intervention(self, intervention_id: int, intervention_data: Dict, user_id: int = None) -> Tuple[bool, str]:
        try:
            current = self.get_intervention_by_id(intervention_id)
            if not current:
                return False, "Intervention non trouvée"

            if any(f in intervention_data for f in ['parts_cost', 'labor_cost', 'travel_cost', 'other_cost']):
                parts = intervention_data.get('parts_cost', current['parts_cost'])
                labor = intervention_data.get('labor_cost', current['labor_cost'])
                travel = intervention_data.get('travel_cost', current['travel_cost'])
                other = intervention_data.get('other_cost', current['other_cost'])
                intervention_data['total_cost'] = parts + labor + travel + other

            if 'start_date' in intervention_data and 'completion_date' in intervention_data:
                try:
                    start = datetime.fromisoformat(intervention_data['start_date'].replace('Z', ''))
                    end = datetime.fromisoformat(intervention_data['completion_date'].replace('Z', ''))
                    duration = (end - start).total_seconds() / 3600
                    intervention_data['actual_duration'] = duration
                except:
                    pass

            updates = []
            params = []
            updateable_fields = [
                'title', 'description', 'type', 'priority', 'status',
                'technician_id', 'supervisor_id', 'assignment_date',
                'start_date', 'pause_date', 'resume_date',
                'completion_date', 'closing_date', 'due_date',
                'estimated_duration', 'actual_duration', 'downtime_hours',
                'cause', 'solution', 'observations', 'work_performed',
                'parts_used', 'parts_cost', 'labor_cost', 'travel_cost',
                'other_cost', 'total_cost', 'satisfaction_score',
                'satisfaction_comment', 'requires_followup', 'followup_id',
                'is_urgent', 'is_planned', 'is_preventive', 'is_corrective',
                'is_warranty', 'is_billed', 'invoice_number',
                'invoice_date', 'invoice_amount'
            ]
            for field in updateable_fields:
                if field in intervention_data:
                    updates.append(f"{field} = ?")
                    params.append(intervention_data[field])

            if not updates:
                return False, "Aucune donnée à mettre à jour"

            if intervention_data.get('status') == 'Terminée' and not current.get('completion_date'):
                updates.append("completion_date = CURRENT_TIMESTAMP")
            if intervention_data.get('status') == 'Fermée' and not current.get('closing_date'):
                updates.append("closing_date = CURRENT_TIMESTAMP")

            query = f"UPDATE interventions SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?"
            params.append(user_id)
            params.append(intervention_id)
            self.db.execute_update(query, tuple(params))

            if intervention_data.get('status') == 'Terminée' and current.get('is_preventive'):
                self.update_asset_maintenance_date(
                    current['asset_id'],
                    intervention_data.get('completion_date') or datetime.now().isoformat()
                )

            changes = {}
            for field in intervention_data:
                if field in current and str(current[field]) != str(intervention_data[field]):
                    changes[field] = {'old': current[field], 'new': intervention_data[field]}

            if changes:
                self.log_action(user_id, 'update', 'intervention', intervention_id,
                              old_values=json.dumps({k: v['old'] for k, v in changes.items()}),
                              new_values=json.dumps({k: v['new'] for k, v in changes.items()}))

            if 'status' in changes:
                self.create_notification(
                    current['technician_id'] or current['requester_id'],
                    f"Statut d'intervention modifié",
                    f"L'intervention '{current['title']}' est maintenant {intervention_data['status']}",
                    f"/interventions/{intervention_id}"
                )

            logger.info(f"Intervention mise à jour: {intervention_id}")
            return True, "Intervention mise à jour"
        except Exception as e:
            logger.error(f"Erreur mise à jour intervention: {e}")
            return False, str(e)

    def delete_intervention(self, intervention_id: int, user_id: int = None) -> Tuple[bool, str]:
        try:
            result = self.db.execute_delete(
                "DELETE FROM interventions WHERE id = ?",
                (intervention_id,)
            )
            if result > 0:
                self.log_action(user_id, 'delete', 'intervention', intervention_id)
                logger.info(f"Intervention supprimée: {intervention_id}")
                return True, "Intervention supprimée"
            return False, "Intervention non trouvée"
        except Exception as e:
            logger.error(f"Erreur suppression intervention: {e}")
            return False, str(e)

    def get_intervention_by_id(self, intervention_id: int) -> Optional[Dict]:
        query = """
            SELECT i.*, 
                   a.code as asset_code, a.name as asset_name,
                   r.first_name || ' ' || r.last_name as requester_name,
                   t.first_name || ' ' || t.last_name as technician_name,
                   s.first_name || ' ' || s.last_name as supervisor_name
            FROM interventions i
            LEFT JOIN assets a ON i.asset_id = a.id
            LEFT JOIN users r ON i.requester_id = r.id
            LEFT JOIN users t ON i.technician_id = t.id
            LEFT JOIN users s ON i.supervisor_id = s.id
            WHERE i.id = ?
        """
        result = self.db.execute_query(query, (intervention_id,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_intervention_by_number(self, number: str) -> Optional[Dict]:
        query = "SELECT * FROM interventions WHERE number = ?"
        result = self.db.execute_query(query, (number,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_all_interventions(self, filters: Dict = None) -> pd.DataFrame:
        query = """
            SELECT i.*, 
                   a.code as asset_code, a.name as asset_name,
                   u.first_name || ' ' || u.last_name as technician_name,
                   CASE 
                       WHEN i.closing_date IS NOT NULL 
                       THEN julianday(i.closing_date) - julianday(i.opening_date)
                       ELSE julianday('now') - julianday(i.opening_date)
                   END as days_open
            FROM interventions i
            LEFT JOIN assets a ON i.asset_id = a.id
            LEFT JOIN users u ON i.technician_id = u.id
            WHERE 1=1
        """
        params = []
        if filters:
            if filters.get('status'):
                if isinstance(filters['status'], list):
                    placeholders = ','.join(['?'] * len(filters['status']))
                    query += f" AND i.status IN ({placeholders})"
                    params.extend(filters['status'])
                else:
                    query += " AND i.status = ?"
                    params.append(filters['status'])
            if filters.get('priority'):
                query += " AND i.priority = ?"
                params.append(filters['priority'])
            if filters.get('type'):
                query += " AND i.type = ?"
                params.append(filters['type'])
            if filters.get('technician_id'):
                query += " AND i.technician_id = ?"
                params.append(filters['technician_id'])
            if filters.get('asset_id'):
                query += " AND i.asset_id = ?"
                params.append(filters['asset_id'])
            if filters.get('requester_id'):
                query += " AND i.requester_id = ?"
                params.append(filters['requester_id'])
            if filters.get('date_debut'):
                query += " AND date(i.opening_date) >= ?"
                params.append(filters['date_debut'])
            if filters.get('date_fin'):
                query += " AND date(i.opening_date) <= ?"
                params.append(filters['date_fin'])
            if filters.get('is_urgent'):
                query += " AND i.is_urgent = 1"
            if filters.get('is_planned'):
                query += " AND i.is_planned = 1"
            if filters.get('is_preventive'):
                query += " AND i.is_preventive = 1"
            if filters.get('is_corrective'):
                query += " AND i.is_corrective = 1"
            if filters.get('is_warranty'):
                query += " AND i.is_warranty = 1"
            if filters.get('search'):
                search = f"%{filters['search']}%"
                query += """ AND (i.number LIKE ? OR i.title LIKE ? OR 
                              i.description LIKE ?)"""
                params.extend([search, search, search])
        query += " ORDER BY i.opening_date DESC"
        return self.db.execute_query(query, tuple(params) if params else None)

    def get_open_interventions(self) -> pd.DataFrame:
        return self.get_all_interventions({
            'status': ['Ouverte', 'Assignée', 'En cours', 'En pause']
        })

    def get_urgent_interventions(self) -> pd.DataFrame:
        return self.get_all_interventions({
            'priority': 'Urgente',
            'status': ['Ouverte', 'Assignée', 'En cours']
        })

    def get_interventions_by_technician(self, technician_id: int, status: str = None) -> pd.DataFrame:
        filters = {'technician_id': technician_id}
        if status:
            filters['status'] = status
        return self.get_all_interventions(filters)

    def get_interventions_by_asset(self, asset_id: int) -> pd.DataFrame:
        return self.get_all_interventions({'asset_id': asset_id})

    def assign_technician(self, intervention_id: int, technician_id: int, user_id: int = None) -> Tuple[bool, str]:
        return self.update_intervention(intervention_id, {
            'technician_id': technician_id,
            'status': 'Assignée',
            'assignment_date': datetime.now().isoformat()
        }, user_id)

    def start_intervention(self, intervention_id: int, user_id: int = None) -> Tuple[bool, str]:
        return self.update_intervention(intervention_id, {
            'status': 'En cours',
            'start_date': datetime.now().isoformat()
        }, user_id)

    def pause_intervention(self, intervention_id: int, user_id: int = None) -> Tuple[bool, str]:
        return self.update_intervention(intervention_id, {
            'status': 'En pause',
            'pause_date': datetime.now().isoformat()
        }, user_id)

    def resume_intervention(self, intervention_id: int, user_id: int = None) -> Tuple[bool, str]:
        return self.update_intervention(intervention_id, {
            'status': 'En cours',
            'resume_date': datetime.now().isoformat()
        }, user_id)

    def complete_intervention(self, intervention_id: int, data: Dict, user_id: int = None) -> Tuple[bool, str]:
        data['status'] = 'Terminée'
        data['completion_date'] = datetime.now().isoformat()
        current = self.get_intervention_by_id(intervention_id)
        if current and current.get('start_date'):
            try:
                start = datetime.fromisoformat(current['start_date'].replace('Z', ''))
                end = datetime.now()
                data['actual_duration'] = (end - start).total_seconds() / 3600
            except:
                pass
        return self.update_intervention(intervention_id, data, user_id)

    def close_intervention(self, intervention_id: int, satisfaction_score: int = None,
                          satisfaction_comment: str = None, user_id: int = None) -> Tuple[bool, str]:
        data = {
            'status': 'Fermée',
            'closing_date': datetime.now().isoformat()
        }
        if satisfaction_score:
            data['satisfaction_score'] = satisfaction_score
        if satisfaction_comment:
            data['satisfaction_comment'] = satisfaction_comment
        return self.update_intervention(intervention_id, data, user_id)

    def update_asset_maintenance_date(self, asset_id: int, maintenance_date: str = None):
        if not maintenance_date:
            maintenance_date = datetime.now().isoformat()
        asset = self.db.execute_query(
            "SELECT maintenance_frequency_days FROM assets WHERE id = ?",
            (asset_id,)
        )
        if not asset.empty:
            frequency = asset.iloc[0]['maintenance_frequency_days']
            if frequency:
                try:
                    maint_date = datetime.fromisoformat(maintenance_date).date()
                    next_date = maint_date + timedelta(days=frequency)
                    self.db.execute_update("""
                        UPDATE assets 
                        SET last_maintenance_date = ?,
                            next_maintenance_date = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (maintenance_date, next_date.isoformat(), asset_id))
                except:
                    self.db.execute_update("""
                        UPDATE assets 
                        SET last_maintenance_date = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (maintenance_date, asset_id))
            else:
                self.db.execute_update("""
                    UPDATE assets 
                    SET last_maintenance_date = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (maintenance_date, asset_id))

    def generate_intervention_number(self) -> str:
        year = datetime.now().year
        month = datetime.now().month
        pattern = f"INT-{year}{month:02d}-%"
        result = self.db.execute_query("""
            SELECT number FROM interventions 
            WHERE number LIKE ? 
            ORDER BY number DESC 
            LIMIT 1
        """, (pattern,))
        if not result.empty:
            last_num = result.iloc[0]['number']
            try:
                last_seq = int(last_num.split('-')[-1])
                new_seq = last_seq + 1
            except:
                new_seq = 1
        else:
            new_seq = 1
        return f"INT-{year}{month:02d}-{new_seq:04d}"

    def validate_intervention_data(self, data: Dict) -> Tuple[bool, str]:
        required_fields = ['title', 'type', 'asset_id']
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f"Le champ {field} est requis"
        asset = self.db.execute_query(
            "SELECT id FROM assets WHERE id = ? AND is_deleted = 0",
            (data['asset_id'],)
        )
        if asset.empty:
            return False, "Équipement invalide"
        if 'due_date' in data and data['due_date']:
            try:
                due = datetime.fromisoformat(data['due_date'].replace('Z', '+00:00'))
                if due < datetime.now():
                    return False, "La date d'échéance ne peut pas être dans le passé"
            except:
                return False, "Format de date invalide pour due_date"
        numeric_fields = ['estimated_duration', 'parts_cost', 'labor_cost',
                         'travel_cost', 'other_cost']
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    float(data[field])
                except:
                    return False, f"{field} doit être un nombre"
        return True, "Données valides"

    def get_intervention_stats(self, period_days: int = 30) -> Dict:
        stats = {}
        total = self.db.execute_query("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status = 'Fermée' THEN 1 ELSE 0 END) as closed,
                   AVG(COALESCE(actual_duration, 0)) as avg_duration,
                   AVG(COALESCE(satisfaction_score, 0)) as avg_satisfaction,
                   SUM(total_cost) as total_cost
            FROM interventions
            WHERE opening_date >= datetime('now', ?)
        """, (f'-{period_days} days',))
        if not total.empty:
            stats['total'] = int(total['total'].iloc[0])
            stats['closed'] = int(total['closed'].iloc[0])
            stats['avg_duration'] = float(total['avg_duration'].iloc[0])
            stats['avg_satisfaction'] = float(total['avg_satisfaction'].iloc[0])
            stats['total_cost'] = float(total['total_cost'].iloc[0]) if total['total_cost'].iloc[0] else 0
            stats['completion_rate'] = round(
                (stats['closed'] / stats['total'] * 100) if stats['total'] > 0 else 0, 1
            )
        by_priority = self.db.execute_query("""
            SELECT priority, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM interventions
            WHERE opening_date >= datetime('now', ?)
            GROUP BY priority
            ORDER BY count DESC
        """, (f'-{period_days} days',))
        stats['by_priority'] = by_priority.to_dict('records') if not by_priority.empty else []
        by_type = self.db.execute_query("""
            SELECT type, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM interventions
            WHERE opening_date >= datetime('now', ?)
            GROUP BY type
            ORDER BY count DESC
        """, (f'-{period_days} days',))
        stats['by_type'] = by_type.to_dict('records') if not by_type.empty else []
        by_status = self.db.execute_query("""
            SELECT status, COUNT(*) as count
            FROM interventions
            WHERE opening_date >= datetime('now', ?)
            GROUP BY status
            ORDER BY count DESC
        """, (f'-{period_days} days',))
        stats['by_status'] = by_status.to_dict('records') if not by_status.empty else []
        daily = self.db.execute_query("""
            SELECT date(opening_date) as date,
                   COUNT(*) as count
            FROM interventions
            WHERE opening_date >= datetime('now', ?)
            GROUP BY date(opening_date)
            ORDER BY date
        """, (f'-{period_days} days',))
        stats['daily'] = daily.to_dict('records') if not daily.empty else []
        return stats

    def create_notification(self, user_id: int, title: str, message: str, link: str = None):
        query = """
            INSERT INTO notifications (user_id, type, title, message, link)
            VALUES (?, 'intervention', ?, ?, ?)
        """
        self.db.execute_insert(query, (user_id, title, message, link))

    def log_action(self, user_id: int, action: str, entity_type: str, entity_id: int,
                  old_values: str = None, new_values: str = None):
        query = """
            INSERT INTO histories (entity_type, entity_id, action, user_id, old_values, new_values)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (entity_type, entity_id, action, user_id, old_values, new_values))

# =============================================================================
# STOCK MANAGER (corrigé)
# =============================================================================

class StockManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_part(self, part_data: Dict, user_id: int = None) -> Tuple[bool, Union[int, str]]:
        try:
            if 'code' not in part_data or not part_data['code']:
                part_data['code'] = self.generate_part_code(part_data.get('category', 'PRT'))
            if 'barcode' not in part_data or not part_data['barcode']:
                part_data['barcode'] = self.generate_barcode()
            part_data['stock_value'] = part_data.get('quantity', 0) * part_data.get('unit_price', 0)

            is_valid, message = self.validate_part_data(part_data)
            if not is_valid:
                return False, message

            query = """
                INSERT INTO spare_parts (
                    code, name, description, category, subcategory,
                    brand, model, supplier_id, supplier_code,
                    manufacturer_id, manufacturer_code, barcode, qr_code, rfid_tag,
                    unit, unit_price, purchase_price, selling_price, vat_rate,
                    quantity, min_quantity, max_quantity, reorder_point, reorder_quantity,
                    location, warehouse, aisle, rack, bin,
                    stock_value, last_purchase_date, last_sale_date, last_inventory_date,
                    notes, is_active, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                part_data.get('code'),
                part_data.get('name'),
                part_data.get('description'),
                part_data.get('category'),
                part_data.get('subcategory'),
                part_data.get('brand'),
                part_data.get('model'),
                part_data.get('supplier_id'),
                part_data.get('supplier_code'),
                part_data.get('manufacturer_id'),
                part_data.get('manufacturer_code'),
                part_data.get('barcode'),
                part_data.get('qr_code'),
                part_data.get('rfid_tag'),
                part_data.get('unit', 'pièce'),
                part_data.get('unit_price', 0),
                part_data.get('purchase_price', 0),
                part_data.get('selling_price', 0),
                part_data.get('vat_rate', 20),
                part_data.get('quantity', 0),
                part_data.get('min_quantity', 0),
                part_data.get('max_quantity', 100),
                part_data.get('reorder_point', 0),
                part_data.get('reorder_quantity', 0),
                part_data.get('location'),
                part_data.get('warehouse'),
                part_data.get('aisle'),
                part_data.get('rack'),
                part_data.get('bin'),
                part_data.get('stock_value', 0),
                part_data.get('last_purchase_date'),
                part_data.get('last_sale_date'),
                part_data.get('last_inventory_date'),
                part_data.get('notes'),
                1,  # is_active
                user_id
            )
            part_id = self.db.execute_insert(query, params)
            self.log_action(user_id, 'create', 'spare_part', part_id, new_values=json.dumps(part_data))
            logger.info(f"Pièce créée: {part_data['code']}")
            return True, part_id
        except Exception as e:
            logger.error(f"Erreur création pièce: {e}")
            return False, str(e)

    def update_part(self, part_id: int, part_data: Dict, user_id: int = None) -> Tuple[bool, str]:
        try:
            current = self.get_part_by_id(part_id)
            if not current:
                return False, "Pièce non trouvée"

            if 'quantity' in part_data or 'unit_price' in part_data:
                quantity = part_data.get('quantity', current['quantity'])
                unit_price = part_data.get('unit_price', current['unit_price'])
                part_data['stock_value'] = quantity * unit_price

            updates = []
            params = []
            updateable_fields = [
                'name', 'description', 'category', 'subcategory',
                'brand', 'model', 'supplier_id', 'supplier_code',
                'manufacturer_id', 'manufacturer_code', 'barcode',
                'qr_code', 'rfid_tag', 'unit', 'unit_price',
                'purchase_price', 'selling_price', 'vat_rate',
                'quantity', 'min_quantity', 'max_quantity',
                'reorder_point', 'reorder_quantity', 'location',
                'warehouse', 'aisle', 'rack', 'bin', 'stock_value',
                'last_purchase_date', 'last_sale_date', 'last_inventory_date',
                'notes', 'is_active'
            ]
            for field in updateable_fields:
                if field in part_data:
                    updates.append(f"{field} = ?")
                    params.append(part_data[field])

            if not updates:
                return False, "Aucune donnée à mettre à jour"

            query = f"UPDATE spare_parts SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?"
            params.append(user_id)
            params.append(part_id)
            self.db.execute_update(query, tuple(params))

            changes = {}
            for field in part_data:
                if field in current and str(current[field]) != str(part_data[field]):
                    changes[field] = {'old': current[field], 'new': part_data[field]}

            if changes:
                self.log_action(user_id, 'update', 'spare_part', part_id,
                              old_values=json.dumps({k: v['old'] for k, v in changes.items()}),
                              new_values=json.dumps({k: v['new'] for k, v in changes.items()}))

            logger.info(f"Pièce mise à jour: {part_id}")
            return True, "Pièce mise à jour"
        except Exception as e:
            logger.error(f"Erreur mise à jour pièce: {e}")
            return False, str(e)

    def delete_part(self, part_id: int, user_id: int = None) -> Tuple[bool, str]:
        try:
            movements = self.db.execute_query(
                "SELECT COUNT(*) as count FROM stock_movements WHERE part_id = ?",
                (part_id,)
            )
            if not movements.empty and movements.iloc[0]['count'] > 0:
                result = self.db.execute_update(
                    "UPDATE spare_parts SET is_active = 0, is_deleted = 1, updated_at = CURRENT_TIMESTAMP, updated_by = ? WHERE id = ?",
                    (user_id, part_id)
                )
            else:
                result = self.db.execute_delete(
                    "DELETE FROM spare_parts WHERE id = ?",
                    (part_id,)
                )
            if result > 0:
                self.log_action(user_id, 'delete', 'spare_part', part_id)
                logger.info(f"Pièce supprimée: {part_id}")
                return True, "Pièce supprimée"
            return False, "Pièce non trouvée"
        except Exception as e:
            logger.error(f"Erreur suppression pièce: {e}")
            return False, str(e)

    def get_part_by_id(self, part_id: int) -> Optional[Dict]:
        query = """
            SELECT p.*, s.name as supplier_name, m.name as manufacturer_name
            FROM spare_parts p
            LEFT JOIN suppliers s ON p.supplier_id = s.id
            LEFT JOIN suppliers m ON p.manufacturer_id = m.id
            WHERE p.id = ? AND p.is_deleted = 0
        """
        result = self.db.execute_query(query, (part_id,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_part_by_code(self, code: str) -> Optional[Dict]:
        query = "SELECT * FROM spare_parts WHERE code = ? AND is_deleted = 0"
        result = self.db.execute_query(query, (code,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_part_by_barcode(self, barcode: str) -> Optional[Dict]:
        query = "SELECT * FROM spare_parts WHERE barcode = ? AND is_deleted = 0"
        result = self.db.execute_query(query, (barcode,))
        if not result.empty:
            return result.iloc[0].to_dict()
        return None

    def get_all_parts(self, filters: Dict = None) -> pd.DataFrame:
        query = """
            SELECT p.*, s.name as supplier_name,
                   CASE 
                       WHEN p.quantity <= 0 THEN 'Rupture'
                       WHEN p.quantity <= p.min_quantity THEN 'Critique'
                       WHEN p.quantity <= p.reorder_point THEN 'Alerte'
                       WHEN p.quantity >= p.max_quantity THEN 'Surcharge'
                       ELSE 'Normal'
                   END as stock_status
            FROM spare_parts p
            LEFT JOIN suppliers s ON p.supplier_id = s.id
            WHERE p.is_deleted = 0
        """
        params = []
        if filters:
            if filters.get('category'):
                query += " AND p.category = ?"
                params.append(filters['category'])
            if filters.get('supplier_id'):
                query += " AND p.supplier_id = ?"
                params.append(filters['supplier_id'])
            if filters.get('location'):
                query += " AND p.location LIKE ?"
                params.append(f'%{filters["location"]}%')
            if filters.get('stock_status'):
                if filters['stock_status'] == 'Rupture':
                    query += " AND p.quantity <= 0"
                elif filters['stock_status'] == 'Critique':
                    query += " AND p.quantity <= p.min_quantity AND p.quantity > 0"
                elif filters['stock_status'] == 'Alerte':
                    query += " AND p.quantity <= p.reorder_point AND p.quantity > p.min_quantity"
                elif filters['stock_status'] == 'Surcharge':
                    query += " AND p.quantity >= p.max_quantity"
            if filters.get('is_active') is not None:
                query += " AND p.is_active = ?"
                params.append(1 if filters['is_active'] else 0)
            if filters.get('search'):
                search = f"%{filters['search']}%"
                query += """ AND (p.name LIKE ? OR p.code LIKE ? OR 
                              p.description LIKE ? OR p.barcode LIKE ?)"""
                params.extend([search, search, search, search])
        query += " ORDER BY p.name"
        return self.db.execute_query(query, tuple(params) if params else None)

    def get_low_stock_parts(self) -> pd.DataFrame:
        return self.get_all_parts({'stock_status': 'Alerte'})

    def get_out_of_stock_parts(self) -> pd.DataFrame:
        return self.get_all_parts({'stock_status': 'Rupture'})

    def add_stock_movement(self, movement_data: Dict, user_id: int = None) -> Tuple[bool, Union[int, str]]:
        try:
            part = self.get_part_by_id(movement_data['part_id'])
            if not part:
                return False, "Pièce non trouvée"

            before_qty = part['quantity']
            quantity = movement_data['quantity']

            if movement_data['type'] == 'Sortie':
                if quantity > before_qty:
                    return False, "Quantité insuffisante en stock"
                after_qty = before_qty - quantity
            elif movement_data['type'] == 'Entrée':
                after_qty = before_qty + quantity
            else:
                after_qty = quantity

            unit_price = movement_data.get('unit_price', part['unit_price'])
            total_price = quantity * unit_price

            query = """
                INSERT INTO stock_movements (
                    part_id, type, quantity, before_quantity, after_quantity,
                    unit_price, total_price, reference_type, reference_id,
                    document_number, reason, notes, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                movement_data['part_id'],
                movement_data['type'],
                quantity,
                before_qty,
                after_qty,
                unit_price,
                total_price,
                movement_data.get('reference_type'),
                movement_data.get('reference_id'),
                movement_data.get('document_number'),
                movement_data.get('reason'),
                movement_data.get('notes'),
                user_id
            )
            movement_id = self.db.execute_insert(query, params)

            self.update_part(part['id'], {
                'quantity': after_qty,
                'last_purchase_date' if movement_data['type'] == 'Entrée' else 'last_sale_date': datetime.now().date().isoformat()
            }, user_id)

            self.log_action(user_id, 'create', 'stock_movement', movement_id,
                          new_values=json.dumps(movement_data))

            if after_qty <= part['min_quantity']:
                self.create_stock_alert(part['id'], f"Stock critique: {part['name']}")
            elif after_qty <= part['reorder_point']:
                self.create_stock_alert(part['id'], f"Stock faible: {part['name']}", 'warning')

            logger.info(f"Mouvement de stock ajouté: {movement_id}")
            return True, movement_id
        except Exception as e:
            logger.error(f"Erreur ajout mouvement stock: {e}")
            return False, str(e)

    def get_stock_movements(self, part_id: int = None, limit: int = 100) -> pd.DataFrame:
        query = """
            SELECT m.*, p.name as part_name, p.code as part_code,
                   u.first_name || ' ' || u.last_name as created_by_name
            FROM stock_movements m
            JOIN spare_parts p ON m.part_id = p.id
            LEFT JOIN users u ON m.created_by = u.id
            WHERE 1=1
        """
        params = []
        if part_id:
            query += " AND m.part_id = ?"
            params.append(part_id)
        query += " ORDER BY m.movement_date DESC LIMIT ?"
        params.append(limit)
        return self.db.execute_query(query, tuple(params))

    def get_stock_value(self) -> float:
        result = self.db.execute_query("SELECT SUM(stock_value) as total FROM spare_parts WHERE is_deleted = 0")
        return float(result['total'].iloc[0]) if not result.empty and result['total'].iloc[0] else 0

    def get_stock_stats(self) -> Dict:
        stats = {}
        total = self.db.execute_query("SELECT COUNT(*) as count FROM spare_parts WHERE is_deleted = 0")
        stats['total_articles'] = int(total['count'].iloc[0]) if not total.empty else 0
        stats['valeur_totale'] = self.get_stock_value()
        rupture = self.db.execute_query("""
            SELECT COUNT(*) as count FROM spare_parts 
            WHERE quantity <= 0 AND is_deleted = 0
        """)
        stats['rupture'] = int(rupture['count'].iloc[0]) if not rupture.empty else 0
        critique = self.db.execute_query("""
            SELECT COUNT(*) as count FROM spare_parts 
            WHERE quantity <= min_quantity AND quantity > 0 AND is_deleted = 0
        """)
        stats['critique'] = int(critique['count'].iloc[0]) if not critique.empty else 0
        alerte = self.db.execute_query("""
            SELECT COUNT(*) as count FROM spare_parts 
            WHERE quantity <= reorder_point AND quantity > min_quantity AND is_deleted = 0
        """)
        stats['alerte'] = int(alerte['count'].iloc[0]) if not alerte.empty else 0
        by_category = self.db.execute_query("""
            SELECT category, COUNT(*) as count,
                   SUM(stock_value) as value
            FROM spare_parts
            WHERE is_deleted = 0
            GROUP BY category
        """)
        stats['by_category'] = by_category.to_dict('records') if not by_category.empty else []
        return stats

    def create_stock_alert(self, part_id: int, message: str, severity: str = 'critical'):
        users = self.db.execute_query("""
            SELECT id FROM users 
            WHERE role IN ('stock_manager', 'purchaser', 'admin') 
              AND is_active = 1
        """)
        for _, user in users.iterrows():
            query = """
                INSERT INTO notifications (user_id, type, title, message, link)
                VALUES (?, 'stock', 'Alerte Stock', ?, ?)
            """
            self.db.execute_insert(query, (user['id'], message, f"/stock/{part_id}"))

    def generate_part_code(self, category: str) -> str:
        prefix = category[:3].upper() if category else 'PRT'
        result = self.db.execute_query("""
            SELECT code FROM spare_parts 
            WHERE code LIKE ? 
            ORDER BY code DESC 
            LIMIT 1
        """, (f"{prefix}%",))
        if not result.empty:
            last_code = result.iloc[0]['code']
            try:
                last_num = int(last_code[len(prefix):])
                new_num = last_num + 1
            except:
                new_num = 1
        else:
            new_num = 1
        return f"{prefix}{new_num:05d}"

    def generate_barcode(self) -> str:
        import random
        while True:
            code = ''.join([str(random.randint(0, 9)) for _ in range(12)])
            total = 0
            for i, digit in enumerate(code):
                if i % 2 == 0:
                    total += int(digit)
                else:
                    total += int(digit) * 3
            check_digit = (10 - (total % 10)) % 10
            barcode = code + str(check_digit)
            existing = self.db.execute_query(
                "SELECT id FROM spare_parts WHERE barcode = ? UNION SELECT id FROM assets WHERE barcode = ?",
                (barcode, barcode)
            )
            if existing.empty:
                return barcode

    def validate_part_data(self, data: Dict) -> Tuple[bool, str]:
        required_fields = ['name']
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f"Le champ {field} est requis"
        numeric_fields = ['quantity', 'min_quantity', 'max_quantity',
                         'reorder_point', 'reorder_quantity', 'unit_price',
                         'purchase_price', 'selling_price', 'vat_rate']
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    float(data[field])
                except:
                    return False, f"{field} doit être un nombre"
        if data.get('min_quantity') and data.get('max_quantity'):
            if data['min_quantity'] > data['max_quantity']:
                return False, "La quantité minimum ne peut pas être supérieure au maximum"
        return True, "Données valides"

    def log_action(self, user_id: int, action: str, entity_type: str, entity_id: int,
                  old_values: str = None, new_values: str = None):
        query = """
            INSERT INTO histories (entity_type, entity_id, action, user_id, old_values, new_values)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.db.execute_insert(query, (entity_type, entity_id, action, user_id, old_values, new_values))

    def export_stock(self, format: str = 'csv', filters: Dict = None) -> Union[str, bytes]:
        parts = self.get_all_parts(filters)
        if format == 'csv':
            return parts.to_csv(index=False, encoding='utf-8-sig')
        elif format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                parts.to_excel(writer, sheet_name='Stock', index=False)
                worksheet = writer.sheets['Stock']
                for i, col in enumerate(parts.columns):
                    max_len = max(parts[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, min(max_len, 50))
            return output.getvalue()
        elif format == 'json':
            return parts.to_json(orient='records', indent=2, force_ascii=False)
        else:
            return None

# =============================================================================
# DASHBOARD RENDERER (corrigé)
# =============================================================================

class DashboardRenderer:
    def __init__(self, db_manager: DatabaseManager, auth_manager: AuthenticationManager,
                 asset_manager: AssetManager, intervention_manager: InterventionManager,
                 stock_manager: StockManager):
        self.db = db_manager
        self.auth = auth_manager
        self.assets = asset_manager
        self.interventions = intervention_manager
        self.stock = stock_manager

    def render_kpi_cards(self):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            asset_stats = self.assets.get_asset_stats()
            st.metric(
                "Équipements",
                asset_stats.get('total', 0),
                f"{asset_stats.get('actifs', 0)} actifs",
                delta_color="normal"
            )
            with st.expander("Détails"):
                st.write(f"En maintenance: {asset_stats.get('en_maintenance', 0)}")
                st.write(f"Hors service: {asset_stats.get('hors_service', 0)}")
                st.write(f"Maintenance due: {asset_stats.get('maintenance_due', 0)}")
        with col2:
            interv_stats = self.interventions.get_intervention_stats(7)
            st.metric(
                "Interventions (7j)",
                interv_stats.get('total', 0),
                f"{interv_stats.get('completion_rate', 0)}% complétées"
            )
            with st.expander("Détails"):
                st.write(f"En cours: {len(self.interventions.get_open_interventions())}")
                st.write(f"Urgentes: {len(self.interventions.get_urgent_interventions())}")
                st.write(f"Coût total: {interv_stats.get('total_cost', 0):,.0f} €")
        with col3:
            stock_stats = self.stock.get_stock_stats()
            st.metric(
                "Stock",
                stock_stats.get('total_articles', 0),
                f"{stock_stats.get('valeur_totale', 0):,.0f} €"
            )
            with st.expander("Détails"):
                st.write(f"Rupture: {stock_stats.get('rupture', 0)}")
                st.write(f"Critique: {stock_stats.get('critique', 0)}")
                st.write(f"Alerte: {stock_stats.get('alerte', 0)}")
        with col4:
            satisfaction = self.db.execute_query("""
                SELECT AVG(satisfaction_score) as avg_satisfaction
                FROM interventions
                WHERE satisfaction_score IS NOT NULL
                  AND closing_date >= date('now', '-30 days')
            """)
            avg_sat = float(satisfaction['avg_satisfaction'].iloc[0]) if not satisfaction.empty and satisfaction['avg_satisfaction'].iloc[0] else 0
            st.metric(
                "Satisfaction",
                f"{avg_sat:.1f}/10" if avg_sat > 0 else "N/A",
                "30 jours"
            )
            with st.expander("Détails"):
                scores = self.db.execute_query("""
                    SELECT 
                        COUNT(CASE WHEN satisfaction_score >= 8 THEN 1 END) as satisfaits,
                        COUNT(CASE WHEN satisfaction_score BETWEEN 5 AND 7 THEN 1 END) as neutres,
                        COUNT(CASE WHEN satisfaction_score < 5 THEN 1 END) as insatisfaits
                    FROM interventions
                    WHERE satisfaction_score IS NOT NULL
                """)
                if not scores.empty:
                    st.write(f"Satisfaits: {scores['satisfaits'].iloc[0]}")
                    st.write(f"Neutres: {scores['neutres'].iloc[0]}")
                    st.write(f"Insatisfaits: {scores['insatisfaits'].iloc[0]}")

    def render_charts(self):
        col1, col2 = st.columns(2)
        with col1:
            assets_by_status = self.assets.get_assets_by_status()
            if not assets_by_status.empty:
                fig = px.pie(
                    assets_by_status,
                    values='count',
                    names='status',
                    title="Répartition des équipements par statut",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        with col2:
            interv_by_priority = self.db.execute_query("""
                SELECT priority, COUNT(*) as count
                FROM interventions
                WHERE opening_date >= date('now', '-30 days')
                GROUP BY priority
            """)
            if not interv_by_priority.empty:
                colors = {
                    'Basse': 'green',
                    'Normale': 'blue',
                    'Haute': 'orange',
                    'Urgente': 'red',
                    'Critique': 'darkred'
                }
                fig = px.bar(
                    interv_by_priority,
                    x='priority',
                    y='count',
                    title="Interventions par priorité (30 jours)",
                    color='priority',
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        col1, col2 = st.columns(2)
        with col1:
            interv_timeline = self.db.execute_query("""
                SELECT date(opening_date) as date,
                       COUNT(*) as count
                FROM interventions
                WHERE opening_date >= date('now', '-30 days')
                GROUP BY date(opening_date)
                ORDER BY date
            """)
            if not interv_timeline.empty:
                fig = px.line(
                    interv_timeline,
                    x='date',
                    y='count',
                    title="Évolution des interventions (30 jours)",
                    markers=True
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Nombre")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        with col2:
            stock_by_category = self.db.execute_query("""
                SELECT category, SUM(stock_value) as value
                FROM spare_parts
                WHERE is_deleted = 0
                GROUP BY category
                ORDER BY value DESC
                LIMIT 10
            """)
            if not stock_by_category.empty:
                fig = px.bar(
                    stock_by_category,
                    x='category',
                    y='value',
                    title="Valeur du stock par catégorie",
                    text_auto='.2s'
                )
                fig.update_layout(xaxis_title="Catégorie", yaxis_title="Valeur (€)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")

    def render_tables(self):
        tab1, tab2, tab3 = st.tabs(["🔧 Maintenances à venir", "⚠️ Alertes stock", "🆕 Interventions récentes"])
        with tab1:
            maintenances = self.assets.get_assets_due_for_maintenance(30)
            if not maintenances.empty:
                display_df = maintenances[['code', 'name', 'next_maintenance_date', 'responsible_name', 'days_until_due']].copy()
                display_df['days_until_due'] = display_df['days_until_due'].round(0).astype(int)
                display_df.columns = ['Code', 'Équipement', 'Date prévue', 'Responsable', 'Jours restants']

                def color_days(val):
                    if val <= 0:
                        return 'color: red; font-weight: bold'
                    elif val <= 7:
                        return 'color: orange; font-weight: bold'
                    elif val <= 30:
                        return 'color: blue'
                    return ''

                styled_df = display_df.style.map(color_days, subset=['Jours restants'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune maintenance prévue dans les 30 prochains jours")
        with tab2:
            alertes = self.stock.get_low_stock_parts()
            if not alertes.empty:
                display_df = alertes[['code', 'name', 'quantity', 'min_quantity', 'reorder_point', 'supplier_name']].copy()
                display_df.columns = ['Code', 'Pièce', 'Quantité', 'Min', 'Seuil', 'Fournisseur']

                def color_qty(row):
                    if row['Quantité'] <= 0:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Quantité'] <= row['Min']:
                        return ['background-color: #ffe6cc'] * len(row)
                    return [''] * len(row)

                styled_df = display_df.style.apply(color_qty, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune alerte stock")
        with tab3:
            recent = self.interventions.get_all_interventions({'date_debut': (datetime.now() - timedelta(days=7)).date().isoformat()})
            if not recent.empty:
                display_df = recent[['number', 'title', 'asset_name', 'priority', 'status', 'technician_name', 'opening_date']].copy()
                display_df.columns = ['N°', 'Titre', 'Équipement', 'Priorité', 'Statut', 'Technicien', 'Date ouverture']

                def color_priority(val):
                    colors = {
                        'Urgente': 'color: red; font-weight: bold',
                        'Haute': 'color: orange',
                        'Normale': 'color: blue',
                        'Basse': 'color: green'
                    }
                    return colors.get(val, '')

                styled_df = display_df.style.map(color_priority, subset=['Priorité'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune intervention récente")

    def render_activity_timeline(self):
        activities = self.db.execute_query("""
            SELECT 'intervention' as type, number as ref, title, status, opening_date as date
            FROM interventions
            WHERE opening_date >= datetime('now', '-7 days')
            UNION ALL
            SELECT 'stock' as type, code as ref, name as title, 
                   CASE WHEN quantity <= 0 THEN 'Rupture' ELSE 'Mouvement' END as status,
                   updated_at as date
            FROM spare_parts
            WHERE updated_at >= datetime('now', '-7 days')
            ORDER BY date DESC
            LIMIT 20
        """)
        if not activities.empty:
            for _, act in activities.iterrows():
                icon = "🛠️" if act['type'] == 'intervention' else "📦"
                status_color = {
                    'Ouverte': '🔵',
                    'En cours': '🟡',
                    'Terminée': '🟢',
                    'Fermée': '⚪',
                    'Rupture': '🔴',
                    'Mouvement': '🟣'
                }.get(act['status'], '⚪')
                with st.container():
                    col1, col2, col3 = st.columns([1, 8, 2])
                    with col1:
                        st.markdown(f"<h3>{icon}</h3>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**{act['title']}**")
                        st.caption(f"{act['ref']} - {act['type']}")
                    with col3:
                        st.markdown(f"{status_color} {act['status']}")
                        st.caption(act['date'][:10])
                    st.divider()
        else:
            st.info("Aucune activité récente")

# =============================================================================
# MAIN APPLICATION (corrigé)
# =============================================================================

class GMAOApplication:
    def __init__(self):
        self.start_time = time.time()
        self.db_path = "data/gmao.db"
        self.db = DatabaseManager(self.db_path)
        self.auth = AuthenticationManager(self.db)
        self.assets = AssetManager(self.db)
        self.interventions = InterventionManager(self.db)
        self.stock = StockManager(self.db)
        self.dashboard = DashboardRenderer(
            self.db, self.auth, self.assets,
            self.interventions, self.stock
        )
        self.init_session()
        logger.info(f"Application initialisée en {time.time() - self.start_time:.2f}s")

    def init_session(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session_id = None
            st.session_state.page = "login"
            st.session_state.notifications = []
            st.session_state.last_activity = datetime.now()

    def run(self):
        try:
            with st.sidebar:
                self.render_sidebar()
            if not st.session_state.authenticated:
                self.render_login()
            else:
                self.render_main_content()
            self.render_footer()
        except Exception as e:
            logger.error(f"Erreur application: {e}")
            logger.error(traceback.format_exc())
            st.error("Une erreur est survenue")
            if st.checkbox("Afficher les détails"):
                st.exception(e)

    def render_sidebar(self):
        st.image("https://via.placeholder.com/200x80?text=GMAO+Enterprise", use_column_width=True)
        if not st.session_state.authenticated:
            st.markdown("### GMAO Enterprise")
            st.markdown("Version 3.0.0")
            st.markdown("---")
            st.info("Veuillez vous connecter")
            with st.expander("Identifiants de démo"):
                st.markdown("""
                **Admin:** admin / admin123
                **Manager:** manager / manager123
                **Technicien:** tech / tech123
                """)
        else:
            user = st.session_state.user
            st.markdown(f"### 👤 {user['first_name']} {user['last_name']}")
            st.caption(f"@{user['username']} - {user['role']}")
            st.markdown("---")
            menu_items = {
                "🏠 Dashboard": "dashboard",
                "🔧 Équipements": "assets",
                "🛠️ Interventions": "interventions",
                "📦 Stock": "stock",
                "🏭 Fournisseurs": "suppliers",
                "📊 Rapports": "reports",
                "⚙️ Paramètres": "settings"
            }
            for label, page in menu_items.items():
                if st.button(label, use_container_width=True,
                           type="primary" if st.session_state.page == page else "secondary"):
                    st.session_state.page = page
                    st.rerun()
            st.markdown("---")
            notif_count = self.db.execute_query("""
                SELECT COUNT(*) as count FROM notifications
                WHERE user_id = ? AND is_read = 0
            """, (user['id'],))
            if not notif_count.empty and notif_count.iloc[0]['count'] > 0:
                st.warning(f"🔔 {notif_count.iloc[0]['count']} notification(s)")
            if st.button("🚪 Déconnexion", use_container_width=True):
                self.logout()

    def render_login(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 Connexion")
            st.markdown("---")
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur")
                password = st.text_input("Mot de passe", type="password")
                remember = st.checkbox("Se souvenir de moi")
                if st.form_submit_button("Se connecter", use_container_width=True):
                    ip_address = "unknown"
                    user_agent = "unknown"
                    try:
                        result = self.auth.authenticate(username, password, ip_address, user_agent)
                        if result:
                            st.session_state.authenticated = True
                            st.session_state.user = result['user']
                            st.session_state.session_id = result['session_id']
                            st.session_state.page = "dashboard"
                            st.success("Connexion réussie!")
                            st.rerun()
                    except AuthenticationError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error("Erreur lors de la connexion")
            st.markdown("---")
            st.markdown("© 2024 GMAO Enterprise - Tous droits réservés")

    def render_main_content(self):
        pages = {
            "dashboard": self.render_dashboard,
            "assets": self.render_assets,
            "interventions": self.render_interventions,
            "stock": self.render_stock,
            "suppliers": self.render_suppliers,
            "reports": self.render_reports,
            "settings": self.render_settings
        }
        page_func = pages.get(st.session_state.page, self.render_dashboard)
        page_func()

    def render_dashboard(self):
        st.title("🏠 Tableau de bord")
        self.dashboard.render_kpi_cards()
        st.markdown("---")
        self.dashboard.render_charts()
        st.markdown("---")
        self.dashboard.render_tables()
        st.markdown("---")
        with st.expander("📋 Activités récentes", expanded=True):
            self.dashboard.render_activity_timeline()

    def render_assets(self):
        st.title("🔧 Gestion des équipements")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Liste", "➕ Ajouter", "📊 Statistiques", "📈 Maintenances"])
        with tab1:
            self.render_assets_list()
        with tab2:
            self.render_asset_form()
        with tab3:
            self.render_assets_stats()
        with tab4:
            self.render_maintenances()

    def render_assets_list(self):
        with st.expander("🔍 Filtres", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_filter = st.selectbox("Statut", ["Tous"] + [s.value for s in AssetStatus])
            with col2:
                type_filter = st.text_input("Type", placeholder="Filtrer par type")
            with col3:
                location_filter = st.text_input("Localisation", placeholder="Filtrer par lieu")
            with col4:
                search = st.text_input("Recherche", placeholder="Nom, code, série...")

        filters = {}
        if status_filter != "Tous":
            filters['status'] = status_filter
        if type_filter:
            filters['type'] = type_filter
        if location_filter:
            filters['location'] = location_filter
        if search:
            filters['search'] = search

        assets = self.assets.get_all_assets(filters)
        if not assets.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(assets))
            with col2:
                actifs = len(assets[assets['status'] == 'Actif'])
                st.metric("Actifs", actifs)
            with col3:
                maintenance = len(assets[assets['status'] == 'En maintenance'])
                st.metric("En maintenance", maintenance)
            with col4:
                valeur = assets['current_value'].sum()
                st.metric("Valeur totale", f"{valeur:,.0f} €")

            display_cols = ['code', 'name', 'type', 'model', 'location',
                           'status', 'responsible_name', 'next_maintenance_date']
            display_df = assets[display_cols].copy()
            display_df.columns = ['Code', 'Nom', 'Type', 'Modèle', 'Localisation',
                                 'Statut', 'Responsable', 'Prochain entretien']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                selected = st.selectbox(
                    "Sélectionner un équipement",
                    options=assets['code'].tolist(),
                    format_func=lambda x: f"{x} - {assets[assets['code']==x]['name'].iloc[0]}"
                )
            if selected:
                asset = assets[assets['code'] == selected].iloc[0]
                with col2:
                    if st.button("✏️ Modifier", use_container_width=True):
                        st.session_state['edit_asset'] = asset.to_dict()
                        st.rerun()
                with col3:
                    if st.button("🗑️ Supprimer", use_container_width=True):
                        success, msg = self.assets.delete_asset(asset['id'], st.session_state.user['id'])
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("Aucun équipement trouvé")

    def render_asset_form(self):
        editing = 'edit_asset' in st.session_state
        asset = st.session_state.get('edit_asset', {})
        with st.form("asset_form"):
            st.subheader("📝 " + ("Modifier l'équipement" if editing else "Nouvel équipement"))
            col1, col2 = st.columns(2)
            with col1:
                code = st.text_input("Code *", value=asset.get('code', ''), disabled=editing, help="Code unique de l'équipement")
                name = st.text_input("Nom *", value=asset.get('name', ''))
                type_ = st.selectbox("Type *", ["Machine", "Équipement", "Véhicule", "Outil", "Infrastructure"],
                                   index=["Machine", "Équipement", "Véhicule", "Outil", "Infrastructure"].index(asset.get('type', 'Machine')) if asset.get('type') in ["Machine", "Équipement", "Véhicule", "Outil", "Infrastructure"] else 0)
                model = st.text_input("Modèle", value=asset.get('model', ''))
                manufacturer = st.text_input("Fabricant", value=asset.get('manufacturer', ''))
                serial_number = st.text_input("N° de série", value=asset.get('serial_number', ''))
                acq_date = None
                if asset.get('acquisition_date'):
                    try:
                        acq_date = datetime.strptime(asset['acquisition_date'], '%Y-%m-%d').date()
                    except:
                        acq_date = datetime.now().date()
                acquisition_date = st.date_input("Date d'acquisition", value=acq_date or datetime.now().date())
                comm_date = None
                if asset.get('commissioning_date'):
                    try:
                        comm_date = datetime.strptime(asset['commissioning_date'], '%Y-%m-%d').date()
                    except:
                        comm_date = datetime.now().date()
                commissioning_date = st.date_input("Date de mise en service", value=comm_date or datetime.now().date())
                warranty_days = st.number_input("Garantie (jours)", min_value=0, value=int(asset.get('warranty_days', 365)))
            with col2:
                location = st.text_input("Localisation", value=asset.get('location', ''))
                department = st.text_input("Département", value=asset.get('department', ''))
                users = self.auth.get_all_users()
                responsables = {row['id']: row['full_name'] for _, row in users.iterrows()}
                resp_index = 0
                if asset.get('responsible_id') in responsables:
                    resp_index = list(responsables.keys()).index(asset['responsible_id'])
                responsable_id = st.selectbox("Responsable", options=list(responsables.keys()), format_func=lambda x: responsables.get(x, ""), index=resp_index)
                status = st.selectbox("Statut", [s.value for s in AssetStatus],
                                    index=[s.value for s in AssetStatus].index(asset.get('status', 'Actif')) if asset.get('status') in [s.value for s in AssetStatus] else 0)
                purchase_price = st.number_input("Prix d'achat (€)", min_value=0.0, value=float(asset.get('purchase_price', 0)))
                depreciation_rate = st.number_input("Taux d'amortissement (%)", min_value=0.0, max_value=100.0, value=float(asset.get('depreciation_rate', 0)))
                useful_life = st.number_input("Durée de vie (ans)", min_value=1, value=int(asset.get('useful_life_years', 5)))
                maintenance_frequency = st.number_input("Fréquence maintenance (jours)", min_value=1, value=int(asset.get('maintenance_frequency_days', 90)))
                notes = st.text_area("Notes", value=asset.get('notes', ''), height=100)
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col2:
                if editing:
                    if st.form_submit_button("✅ Mettre à jour", use_container_width=True):
                        data = {
                            'code': code, 'name': name, 'type': type_, 'model': model,
                            'manufacturer': manufacturer, 'serial_number': serial_number,
                            'acquisition_date': acquisition_date.isoformat(),
                            'commissioning_date': commissioning_date.isoformat(),
                            'warranty_days': warranty_days, 'location': location,
                            'department': department, 'responsible_id': responsable_id,
                            'status': status, 'purchase_price': purchase_price,
                            'depreciation_rate': depreciation_rate,
                            'useful_life_years': useful_life,
                            'maintenance_frequency_days': maintenance_frequency,
                            'notes': notes
                        }
                        success, msg = self.assets.update_asset(asset['id'], data, st.session_state.user['id'])
                        if success:
                            st.success(msg)
                            del st.session_state['edit_asset']
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    if st.form_submit_button("✅ Ajouter", use_container_width=True):
                        if not name or not type_:
                            st.error("Veuillez remplir tous les champs obligatoires")
                        else:
                            data = {
                                'code': code, 'name': name, 'type': type_, 'model': model,
                                'manufacturer': manufacturer, 'serial_number': serial_number,
                                'acquisition_date': acquisition_date.isoformat(),
                                'commissioning_date': commissioning_date.isoformat(),
                                'warranty_days': warranty_days, 'location': location,
                                'department': department, 'responsible_id': responsable_id,
                                'status': status, 'purchase_price': purchase_price,
                                'depreciation_rate': depreciation_rate,
                                'useful_life_years': useful_life,
                                'maintenance_frequency_days': maintenance_frequency,
                                'notes': notes
                            }
                            success, result = self.assets.create_asset(data, st.session_state.user['id'])
                            if success:
                                st.success("Équipement ajouté avec succès!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Erreur: {result}")
            if editing:
                col1, col2, col3 = st.columns(3)
                with col2:
                    if st.form_submit_button("❌ Annuler", use_container_width=True):
                        del st.session_state['edit_asset']
                        st.rerun()

    def render_assets_stats(self):
        stats = self.assets.get_asset_stats()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total équipements", stats.get('total', 0))
        with col2:
            st.metric("Valeur totale", f"{stats.get('valeur_totale', 0):,.0f} €")
        with col3:
            st.metric("Âge moyen", f"{stats.get('age_moyen', 0):.1f} ans")
        with col4:
            maint_due = stats.get('maintenance_due', 0)
            st.metric("Maintenances dues", maint_due, delta_color="inverse")
        col1, col2 = st.columns(2)
        with col1:
            by_status = self.assets.get_assets_by_status()
            if not by_status.empty:
                fig = px.pie(by_status, values='count', names='status',
                           title="Répartition par statut",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            by_type = self.assets.get_assets_by_type()
            if not by_type.empty:
                fig = px.bar(by_type, x='type', y='count',
                           title="Répartition par type",
                           color='type')
                st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            by_dept = self.assets.get_assets_by_department()
            if not by_dept.empty:
                fig = px.pie(by_dept, values='count', names='department',
                           title="Répartition par département")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            by_resp = self.assets.get_assets_by_responsible()
            if not by_resp.empty:
                fig = px.bar(by_resp, x='responsible', y='count',
                           title="Équipements par responsable")
                st.plotly_chart(fig, use_container_width=True)

    def render_maintenances(self):
        maintenances = self.assets.get_assets_due_for_maintenance(90)
        if not maintenances.empty:
            total = len(maintenances)
            urgent = len(maintenances[maintenances['days_until_due'] <= 7])
            retard = len(maintenances[maintenances['days_until_due'] < 0])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", total)
            with col2:
                st.metric("Urgentes (≤7j)", urgent, delta_color="inverse")
            with col3:
                st.metric("En retard", retard, delta_color="inverse")
            with col4:
                st.metric("Taux de complétion", f"{((total-retard)/total*100):.1f}%" if total > 0 else "0%")

            import calendar
            from calendar import monthcalendar
            today = datetime.now().date()
            current_month = today.month
            current_year = today.year
            cal = monthcalendar(current_year, current_month)
            month_name = calendar.month_name[current_month]
            st.subheader(f"📅 Calendrier des maintenances - {month_name} {current_year}")
            col_headers = st.columns(7)
            for i, day in enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]):
                with col_headers[i]:
                    st.markdown(f"**{day}**")
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    with cols[i]:
                        if day != 0:
                            date_str = f"{current_year}-{current_month:02d}-{day:02d}"
                            day_maint = maintenances[maintenances['next_maintenance_date'] == date_str]
                            if day == today.day:
                                st.markdown(f"**📅 {day}**")
                            else:
                                st.markdown(f"**{day}**")
                            if not day_maint.empty:
                                for _, m in day_maint.iterrows():
                                    color = "🔴" if m['days_until_due'] < 0 else "🟠" if m['days_until_due'] <= 7 else "🟡"
                                    st.caption(f"{color} {m['code']}")

            st.subheader("📋 Liste détaillée")
            display_df = maintenances[['code', 'name', 'next_maintenance_date', 'responsible_name', 'days_until_due']].copy()
            display_df['days_until_due'] = display_df['days_until_due'].round(0).astype(int)
            display_df.columns = ['Code', 'Équipement', 'Date prévue', 'Responsable', 'Jours restants']

            def color_days(val):
                if val < 0:
                    return 'background-color: #ffcccc; color: red; font-weight: bold'
                elif val <= 7:
                    return 'background-color: #fff3cd; color: orange; font-weight: bold'
                elif val <= 30:
                    return 'background-color: #d4edda; color: green'
                return ''

            styled_df = display_df.style.map(color_days, subset=['Jours restants'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            if not maintenances.empty and st.button("📧 Envoyer rappels", use_container_width=True):
                st.success("Rappels envoyés aux responsables")
        else:
            st.info("Aucune maintenance prévue dans les 90 prochains jours")

    def render_interventions(self):
        st.title("🛠️ Gestion des interventions")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Liste", "➕ Nouvelle", "📊 Statistiques", "⏱️ En cours"])
        with tab1:
            self.render_interventions_list()
        with tab2:
            self.render_intervention_form()
        with tab3:
            self.render_interventions_stats()
        with tab4:
            self.render_ongoing_interventions()

    def render_interventions_list(self):
        with st.expander("🔍 Filtres", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_filter = st.multiselect("Statut", [s.value for s in InterventionStatus], default=[])
            with col2:
                priority_filter = st.multiselect("Priorité", [p.value for p in PriorityLevel], default=[])
            with col3:
                date_range = st.date_input("Période", value=(datetime.now() - timedelta(days=30), datetime.now()), key="interv_date_range")
            with col4:
                search = st.text_input("Recherche", placeholder="N°, titre...")

        filters = {}
        if status_filter:
            filters['status'] = status_filter
        if priority_filter:
            filters['priority'] = priority_filter
        if len(date_range) == 2:
            filters['date_debut'] = date_range[0].isoformat()
            filters['date_fin'] = date_range[1].isoformat()
        if search:
            filters['search'] = search

        interventions = self.interventions.get_all_interventions(filters)
        if not interventions.empty:
            total = len(interventions)
            ouvertes = len(interventions[interventions['status'].isin(['Ouverte', 'Assignée', 'En cours'])])
            terminees = len(interventions[interventions['status'] == 'Terminée'])
            cout_total = interventions['total_cost'].sum()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", total)
            with col2:
                st.metric("En cours", ouvertes)
            with col3:
                st.metric("Terminées", terminees)
            with col4:
                st.metric("Coût total", f"{cout_total:,.0f} €")

            display_cols = ['number', 'title', 'asset_name', 'priority', 'status',
                           'technician_name', 'opening_date', 'days_open']
            display_df = interventions[display_cols].copy()
            display_df['days_open'] = display_df['days_open'].round(1)
            display_df.columns = ['N°', 'Titre', 'Équipement', 'Priorité', 'Statut',
                                 'Technicien', 'Date ouverture', 'Jours ouverts']

            def color_priority(val):
                colors = {
                    'Urgente': 'color: red; font-weight: bold',
                    'Haute': 'color: orange',
                    'Normale': 'color: blue',
                    'Basse': 'color: green'
                }
                return colors.get(val, '')

            def color_status(val):
                colors = {
                    'Ouverte': 'background-color: #fff3cd',
                    'Assignée': 'background-color: #cce5ff',
                    'En cours': 'background-color: #d4edda',
                    'Terminée': 'background-color: #d1ecf1',
                    'Fermée': 'background-color: #f8d7da'
                }
                return colors.get(val, '')

            styled_df = display_df.style.map(color_priority, subset=['Priorité']).map(color_status, subset=['Statut'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                selected = st.selectbox(
                    "Sélectionner une intervention",
                    options=interventions['number'].tolist(),
                    format_func=lambda x: f"{x} - {interventions[interventions['number']==x]['title'].iloc[0]}"
                )
            if selected:
                interv = interventions[interventions['number'] == selected].iloc[0]
                with col2:
                    if st.button("✏️ Modifier", use_container_width=True):
                        st.session_state['edit_intervention'] = interv.to_dict()
                        st.rerun()
                with col3:
                    if st.button("🗑️ Supprimer", use_container_width=True):
                        success, msg = self.interventions.delete_intervention(interv['id'], st.session_state.user['id'])
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("Aucune intervention trouvée")

    def render_intervention_form(self):
        editing = 'edit_intervention' in st.session_state
        interv = st.session_state.get('edit_intervention', {})
        with st.form("intervention_form"):
            st.subheader("📝 " + ("Modifier l'intervention" if editing else "Nouvelle intervention"))
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Titre *", value=interv.get('title', ''))
                assets = self.assets.get_all_assets({'is_active': True})
                asset_options = {row['id']: f"{row['code']} - {row['name']}" for _, row in assets.iterrows()}
                asset_id_index = 0
                if interv.get('asset_id') in asset_options:
                    asset_id_index = list(asset_options.keys()).index(interv['asset_id'])
                asset_id = st.selectbox("Équipement *", options=list(asset_options.keys()), format_func=lambda x: asset_options.get(x, ""), index=asset_id_index)
                type_ = st.selectbox("Type *",
                                   ["Dépannage", "Réparation", "Maintenance préventive",
                                    "Inspection", "Installation", "Modification"],
                                   index=["Dépannage", "Réparation", "Maintenance préventive",
                                        "Inspection", "Installation", "Modification"].index(interv.get('type', 'Dépannage')) if interv.get('type') in ["Dépannage", "Réparation", "Maintenance préventive", "Inspection", "Installation", "Modification"] else 0)
                priority = st.selectbox("Priorité",
                                      [p.value for p in PriorityLevel],
                                      index=[p.value for p in PriorityLevel].index(interv.get('priority', 'Normale')) if interv.get('priority') in [p.value for p in PriorityLevel] else 2)
                description = st.text_area("Description", value=interv.get('description', ''), height=100)
                cause = st.text_area("Cause", value=interv.get('cause', ''), height=100)
            with col2:
                users = self.auth.get_all_users()
                user_options = {row['id']: row['full_name'] for _, row in users.iterrows()}
                requester_index = 0
                if interv.get('requester_id') in user_options:
                    requester_index = list(user_options.keys()).index(interv['requester_id'])
                requester_id = st.selectbox("Demandeur", options=list(user_options.keys()), format_func=lambda x: user_options.get(x, ""), index=requester_index)
                tech_users = users[users['role'].isin(['technician', 'supervisor'])]
                tech_options = {row['id']: row['full_name'] for _, row in tech_users.iterrows()}
                tech_index = 0
                if interv.get('technician_id') in tech_options:
                    tech_index = list(tech_options.keys()).index(interv['technician_id'])
                technician_id = st.selectbox("Technicien assigné", options=list(tech_options.keys()), format_func=lambda x: tech_options.get(x, ""), index=tech_index if tech_options else 0)
                due_date = None
                if interv.get('due_date'):
                    try:
                        due_date = datetime.strptime(interv['due_date'], '%Y-%m-%d').date()
                    except:
                        due_date = datetime.now().date() + timedelta(days=7)
                due_date = st.date_input("Date d'échéance", value=due_date or (datetime.now().date() + timedelta(days=7)))
                estimated_duration = st.number_input("Durée estimée (heures)", min_value=0.0, value=float(interv.get('estimated_duration', 2.0)))
                col_a, col_b = st.columns(2)
                with col_a:
                    is_urgent = st.checkbox("Urgent", value=interv.get('is_urgent', False))
                    is_preventive = st.checkbox("Maintenance préventive", value=interv.get('is_preventive', False))
                with col_b:
                    is_planned = st.checkbox("Planifiée", value=interv.get('is_planned', False))
                    is_warranty = st.checkbox("Sous garantie", value=interv.get('is_warranty', False))
                notes = st.text_area("Notes", value=interv.get('observations', ''), height=100)

            if editing:
                st.subheader("Résolution")
                col1, col2 = st.columns(2)
                with col1:
                    actual_duration = st.number_input("Durée réelle (heures)", min_value=0.0, value=float(interv.get('actual_duration', 0)))
                    parts_cost = st.number_input("Coût pièces (€)", min_value=0.0, value=float(interv.get('parts_cost', 0)))
                with col2:
                    labor_cost = st.number_input("Main d'œuvre (€)", min_value=0.0, value=float(interv.get('labor_cost', 0)))
                    other_cost = st.number_input("Autres coûts (€)", min_value=0.0, value=float(interv.get('other_cost', 0)))
                solution = st.text_area("Solution apportée", value=interv.get('solution', ''), height=100)
                work_performed = st.text_area("Travaux réalisés", value=interv.get('work_performed', ''), height=100)
                satisfaction_score = st.slider("Satisfaction client", 0, 10, int(interv.get('satisfaction_score', 0)) if interv.get('satisfaction_score') else 0)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col2:
                if editing:
                    if st.form_submit_button("✅ Mettre à jour", use_container_width=True):
                        data = {
                            'title': title,
                            'asset_id': asset_id,
                            'type': type_,
                            'priority': priority,
                            'description': description,
                            'cause': cause,
                            'requester_id': requester_id,
                            'technician_id': technician_id,
                            'due_date': due_date.isoformat(),
                            'estimated_duration': estimated_duration,
                            'is_urgent': is_urgent,
                            'is_preventive': is_preventive,
                            'is_planned': is_planned,
                            'is_warranty': is_warranty,
                            'observations': notes,
                            'actual_duration': actual_duration,
                            'parts_cost': parts_cost,
                            'labor_cost': labor_cost,
                            'other_cost': other_cost,
                            'solution': solution,
                            'work_performed': work_performed,
                            'satisfaction_score': satisfaction_score
                        }
                        success, msg = self.interventions.update_intervention(interv['id'], data, st.session_state.user['id'])
                        if success:
                            st.success(msg)
                            del st.session_state['edit_intervention']
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    if st.form_submit_button("✅ Créer", use_container_width=True):
                        if not title or not asset_id:
                            st.error("Veuillez remplir tous les champs obligatoires")
                        else:
                            data = {
                                'title': title,
                                'asset_id': asset_id,
                                'type': type_,
                                'priority': priority,
                                'description': description,
                                'cause': cause,
                                'requester_id': requester_id,
                                'technician_id': technician_id,
                                'due_date': due_date.isoformat(),
                                'estimated_duration': estimated_duration,
                                'is_urgent': is_urgent,
                                'is_preventive': is_preventive,
                                'is_planned': is_planned,
                                'is_warranty': is_warranty,
                                'observations': notes
                            }
                            success, result = self.interventions.create_intervention(data, st.session_state.user['id'])
                            if success:
                                st.success("Intervention créée avec succès!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Erreur: {result}")
            if editing:
                col1, col2, col3 = st.columns(3)
                with col2:
                    if st.form_submit_button("❌ Annuler", use_container_width=True):
                        del st.session_state['edit_intervention']
                        st.rerun()

    def render_interventions_stats(self):
        period = st.selectbox("Période", [7, 30, 90, 365], format_func=lambda x: f"{x} jours", index=1)
        stats = self.interventions.get_intervention_stats(period)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", stats.get('total', 0))
        with col2:
            st.metric("Taux complétion", f"{stats.get('completion_rate', 0)}%")
        with col3:
            st.metric("Durée moyenne", f"{stats.get('avg_duration', 0):.1f} h")
        with col4:
            st.metric("Coût total", f"{stats.get('total_cost', 0):,.0f} €")
        col1, col2 = st.columns(2)
        with col1:
            if stats.get('by_priority'):
                df_priority = pd.DataFrame(stats['by_priority'])
                fig = px.pie(df_priority, values='count', names='priority',
                           title="Répartition par priorité",
                           color='priority',
                           color_discrete_map={
                               'Basse': 'green',
                               'Normale': 'blue',
                               'Haute': 'orange',
                               'Urgente': 'red'
                           })
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if stats.get('by_type'):
                df_type = pd.DataFrame(stats['by_type'])
                fig = px.bar(df_type, x='type', y='count',
                           title="Répartition par type",
                           color='type')
                st.plotly_chart(fig, use_container_width=True)
        if stats.get('daily'):
            df_daily = pd.DataFrame(stats['daily'])
            fig = px.line(df_daily, x='date', y='count',
                        title=f"Évolution quotidienne ({period} jours)",
                        markers=True)
            st.plotly_chart(fig, use_container_width=True)
        satisfaction = self.db.execute_query("""
            SELECT 
                strftime('%Y-%m', closing_date) as mois,
                AVG(satisfaction_score) as avg_satisfaction
            FROM interventions
            WHERE satisfaction_score IS NOT NULL
              AND closing_date >= date('now', ?)
            GROUP BY strftime('%Y-%m', closing_date)
            ORDER BY mois
        """, (f'-{period} days',))
        if not satisfaction.empty:
            fig = px.line(satisfaction, x='mois', y='avg_satisfaction',
                        title="Évolution de la satisfaction",
                        range_y=[0, 10])
            st.plotly_chart(fig, use_container_width=True)

    def render_ongoing_interventions(self):
        ongoing = self.interventions.get_all_interventions({
            'status': ['En cours', 'Assignée']
        })
        if not ongoing.empty:
            for _, interv in ongoing.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{interv['number']}** - {interv['title']}")
                        st.caption(f"Équipement: {interv['asset_name']}")
                    with col2:
                        priority_color = {
                            'Urgente': '🔴',
                            'Haute': '🟠',
                            'Normale': '🟡',
                            'Basse': '🟢'
                        }.get(interv['priority'], '⚪')
                        st.markdown(f"{priority_color} {interv['priority']}")
                        st.caption(f"Technicien: {interv['technician_name']}")
                    with col3:
                        if interv['start_date']:
                            try:
                                start = datetime.fromisoformat(interv['start_date'].replace('Z', '+00:00'))
                                elapsed = datetime.now() - start
                                hours = elapsed.total_seconds() / 3600
                                st.markdown(f"⏱️ {hours:.1f} h")
                                if interv['estimated_duration']:
                                    progress = min(hours / interv['estimated_duration'] * 100, 100)
                                    st.progress(progress / 100, text=f"{progress:.0f}%")
                            except:
                                pass
                    with col4:
                        if st.button("▶️", key=f"view_{interv['id']}", help="Voir détails"):
                            st.session_state['view_intervention'] = interv['id']
                            st.rerun()
                    st.divider()
        else:
            st.info("Aucune intervention en cours")

    def render_stock(self):
        st.title("📦 Gestion du stock")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Liste", "➕ Nouvelle pièce", "📊 Statistiques", "📦 Mouvements"])
        with tab1:
            self.render_stock_list()
        with tab2:
            self.render_part_form()
        with tab3:
            self.render_stock_stats()
        with tab4:
            self.render_stock_movements()

    def render_stock_list(self):
        with st.expander("🔍 Filtres", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_filter = st.selectbox("État stock", ["Tous", "Normal", "Alerte", "Critique", "Rupture"])
            with col2:
                category_filter = st.text_input("Catégorie", placeholder="Filtrer par catégorie")
            with col3:
                location_filter = st.text_input("Emplacement", placeholder="Filtrer par lieu")
            with col4:
                search = st.text_input("Recherche", placeholder="Nom, code...")

        filters = {}
        if status_filter != "Tous":
            filters['stock_status'] = status_filter
        if category_filter:
            filters['category'] = category_filter
        if location_filter:
            filters['location'] = location_filter
        if search:
            filters['search'] = search

        parts = self.stock.get_all_parts(filters)
        if not parts.empty:
            valeur_totale = parts['stock_value'].sum()
            articles_critiques = len(parts[parts['stock_status'].isin(['Critique', 'Rupture'])])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total articles", len(parts))
            with col2:
                st.metric("Valeur totale", f"{valeur_totale:,.0f} €")
            with col3:
                st.metric("Articles critiques", articles_critiques)
            with col4:
                taux = (1 - articles_critiques/len(parts)) * 100 if len(parts) > 0 else 100
                st.metric("Taux disponibilité", f"{taux:.1f}%")

            display_cols = ['code', 'name', 'category', 'quantity', 'unit',
                           'min_quantity', 'reorder_point', 'location',
                           'stock_status', 'supplier_name']
            display_df = parts[display_cols].copy()
            display_df.columns = ['Code', 'Nom', 'Catégorie', 'Qté', 'Unité',
                                 'Min', 'Seuil', 'Emplacement', 'Statut', 'Fournisseur']

            def color_status(val):
                colors = {
                    'Rupture': 'background-color: #f8d7da; color: #721c24',
                    'Critique': 'background-color: #fff3cd; color: #856404',
                    'Alerte': 'background-color: #fff3cd; color: #856404',
                    'Normal': 'background-color: #d4edda; color: #155724',
                    'Surcharge': 'background-color: #cce5ff; color: #004085'
                }
                return colors.get(val, '')

            styled_df = display_df.style.map(color_status, subset=['Statut'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                selected = st.selectbox(
                    "Sélectionner une pièce",
                    options=parts['code'].tolist(),
                    format_func=lambda x: f"{x} - {parts[parts['code']==x]['name'].iloc[0]}"
                )
            if selected:
                part = parts[parts['code'] == selected].iloc[0]
                with col2:
                    if st.button("✏️ Modifier", use_container_width=True):
                        st.session_state['edit_part'] = part.to_dict()
                        st.rerun()
                with col3:
                    if st.button("➕ Mouvement", use_container_width=True):
                        st.session_state['add_movement'] = part['id']
                        st.rerun()
        else:
            st.info("Aucune pièce trouvée")

    def render_part_form(self):
        editing = 'edit_part' in st.session_state
        part = st.session_state.get('edit_part', {})
        with st.form("part_form"):
            st.subheader("📝 " + ("Modifier la pièce" if editing else "Nouvelle pièce"))
            col1, col2 = st.columns(2)
            with col1:
                code = st.text_input("Code *", value=part.get('code', ''), disabled=editing)
                name = st.text_input("Nom *", value=part.get('name', ''))
                category = st.text_input("Catégorie", value=part.get('category', ''))
                subcategory = st.text_input("Sous-catégorie", value=part.get('subcategory', ''))
                brand = st.text_input("Marque", value=part.get('brand', ''))
                model = st.text_input("Modèle", value=part.get('model', ''))
                description = st.text_area("Description", value=part.get('description', ''), height=100)
            with col2:
                suppliers = self.db.execute_query("SELECT id, name FROM suppliers WHERE is_active = 1")
                supplier_options = {row['id']: row['name'] for _, row in suppliers.iterrows()}
                supplier_index = 0
                if part.get('supplier_id') in supplier_options:
                    supplier_index = list(supplier_options.keys()).index(part['supplier_id'])
                supplier_id = st.selectbox("Fournisseur", options=list(supplier_options.keys()), format_func=lambda x: supplier_options.get(x, ""), index=supplier_index)
                unit = st.selectbox("Unité",
                                  ["pièce", "mètre", "kilogramme", "litre", "boîte", "rouleau"],
                                  index=["pièce", "mètre", "kilogramme", "litre", "boîte", "rouleau"].index(part.get('unit', 'pièce')) if part.get('unit') in ["pièce", "mètre", "kilogramme", "litre", "boîte", "rouleau"] else 0)
                unit_price = st.number_input("Prix unitaire (€)", min_value=0.0, value=float(part.get('unit_price', 0)))
                quantity = st.number_input("Quantité", min_value=0, value=int(part.get('quantity', 0)))
                min_quantity = st.number_input("Quantité minimum", min_value=0, value=int(part.get('min_quantity', 5)))
                max_quantity = st.number_input("Quantité maximum", min_value=0, value=int(part.get('max_quantity', 100)))
                reorder_point = st.number_input("Seuil de réapprovisionnement", min_value=0, value=int(part.get('reorder_point', 10)))
                location = st.text_input("Emplacement", value=part.get('location', ''))
                warehouse = st.text_input("Entrepôt", value=part.get('warehouse', ''))
                bin_location = st.text_input("Casier", value=part.get('bin', ''))
                notes = st.text_area("Notes", value=part.get('notes', ''), height=100)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col2:
                if editing:
                    if st.form_submit_button("✅ Mettre à jour", use_container_width=True):
                        data = {
                            'code': code,
                            'name': name,
                            'category': category,
                            'subcategory': subcategory,
                            'brand': brand,
                            'model': model,
                            'description': description,
                            'supplier_id': supplier_id,
                            'unit': unit,
                            'unit_price': unit_price,
                            'quantity': quantity,
                            'min_quantity': min_quantity,
                            'max_quantity': max_quantity,
                            'reorder_point': reorder_point,
                            'location': location,
                            'warehouse': warehouse,
                            'bin': bin_location,
                            'notes': notes
                        }
                        success, msg = self.stock.update_part(part['id'], data, st.session_state.user['id'])
                        if success:
                            st.success(msg)
                            del st.session_state['edit_part']
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    if st.form_submit_button("✅ Ajouter", use_container_width=True):
                        if not name:
                            st.error("Veuillez remplir tous les champs obligatoires")
                        else:
                            data = {
                                'code': code,
                                'name': name,
                                'category': category,
                                'subcategory': subcategory,
                                'brand': brand,
                                'model': model,
                                'description': description,
                                'supplier_id': supplier_id,
                                'unit': unit,
                                'unit_price': unit_price,
                                'quantity': quantity,
                                'min_quantity': min_quantity,
                                'max_quantity': max_quantity,
                                'reorder_point': reorder_point,
                                'location': location,
                                'warehouse': warehouse,
                                'bin': bin_location,
                                'notes': notes
                            }
                            success, result = self.stock.create_part(data, st.session_state.user['id'])
                            if success:
                                st.success("Pièce ajoutée avec succès!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Erreur: {result}")
            if editing:
                col1, col2, col3 = st.columns(3)
                with col2:
                    if st.form_submit_button("❌ Annuler", use_container_width=True):
                        del st.session_state['edit_part']
                        st.rerun()

    def render_stock_stats(self):
        stats = self.stock.get_stock_stats()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total articles", stats.get('total_articles', 0))
        with col2:
            st.metric("Valeur totale", f"{stats.get('valeur_totale', 0):,.0f} €")
        with col3:
            st.metric("Rupture", stats.get('rupture', 0))
        with col4:
            st.metric("Stock critique", stats.get('critique', 0))

        col1, col2 = st.columns(2)
        with col1:
            stock_status = pd.DataFrame([
                {'statut': 'Normal', 'count': stats.get('total_articles', 0) - stats.get('critique', 0) - stats.get('rupture', 0) - stats.get('alerte', 0)},
                {'statut': 'Alerte', 'count': stats.get('alerte', 0)},
                {'statut': 'Critique', 'count': stats.get('critique', 0)},
                {'statut': 'Rupture', 'count': stats.get('rupture', 0)}
            ])
            fig = px.pie(stock_status, values='count', names='statut',
                       title="État du stock",
                       color='statut',
                       color_discrete_map={
                           'Normal': 'green',
                           'Alerte': 'orange',
                           'Critique': 'red',
                           'Rupture': 'darkred'
                       })
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if stats.get('by_category'):
                df_cat = pd.DataFrame(stats['by_category'])
                fig = px.bar(df_cat, x='category', y='value',
                           title="Valeur du stock par catégorie",
                           text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)

    def render_stock_movements(self):
        if 'add_movement' in st.session_state:
            part_id = st.session_state['add_movement']
            part = self.stock.get_part_by_id(part_id)
            if part:
                with st.form("movement_form"):
                    st.subheader(f"➕ Mouvement pour {part['name']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        movement_type = st.selectbox("Type", ["Entrée", "Sortie", "Ajustement"])
                        quantity = st.number_input("Quantité", min_value=1, value=1)
                    with col2:
                        reason = st.text_input("Raison", placeholder="Ex: Réception commande, Utilisation...")
                        reference = st.text_input("Référence", placeholder="N° commande, intervention...")
                    notes = st.text_area("Notes", height=100)
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        if st.form_submit_button("✅ Valider", use_container_width=True):
                            data = {
                                'part_id': part_id,
                                'type': movement_type,
                                'quantity': quantity if movement_type == 'Entrée' else -quantity,
                                'reason': reason,
                                'document_number': reference,
                                'notes': notes
                            }
                            success, msg = self.stock.add_stock_movement(data, st.session_state.user['id'])
                            if success:
                                st.success("Mouvement enregistré")
                                del st.session_state['add_movement']
                                st.rerun()
                            else:
                                st.error(msg)
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        if st.form_submit_button("❌ Annuler", use_container_width=True):
                            del st.session_state['add_movement']
                            st.rerun()

        st.subheader("Historique des mouvements")
        movements = self.stock.get_stock_movements(limit=50)
        if not movements.empty:
            display_df = movements[['movement_date', 'type', 'part_name', 'part_code',
                                   'quantity', 'before_quantity', 'after_quantity',
                                   'reason', 'created_by_name']].copy()
            display_df.columns = ['Date', 'Type', 'Pièce', 'Code', 'Qté',
                                 'Avant', 'Après', 'Raison', 'Créé par']

            def color_type(val):
                colors = {
                    'Entrée': 'color: green',
                    'Sortie': 'color: red',
                    'Ajustement': 'color: orange'
                }
                return colors.get(val, '')

            styled_df = display_df.style.map(color_type, subset=['Type'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun mouvement enregistré")

    def render_suppliers(self):
        st.title("🏭 Gestion des fournisseurs")
        st.info("Page en cours de développement...")

    def render_reports(self):
        st.title("📊 Rapports")
        report_type = st.selectbox(
            "Type de rapport",
            ["Inventaire équipements", "Interventions", "Mouvements de stock",
             "Coûts de maintenance", "Performance techniciens"]
        )
        col1, col2 = st.columns(2)
        with col1:
            date_debut = st.date_input("Date début", value=datetime.now() - timedelta(days=30))
        with col2:
            date_fin = st.date_input("Date fin", value=datetime.now())
        format_ = st.selectbox("Format d'export", ["Excel", "PDF", "CSV"])
        if st.button("📥 Générer le rapport", use_container_width=True):
            with st.spinner("Génération du rapport..."):
                time.sleep(2)
                st.success("Rapport généré avec succès!")
                st.download_button(
                    "📥 Télécharger",
                    data="Rapport simulé".encode(),
                    file_name=f"rapport_{report_type}_{date_debut}_{date_fin}.{format_.lower()}",
                    mime="application/octet-stream"
                )

    def render_settings(self):
        st.title("⚙️ Paramètres")
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Profil", "🔐 Sécurité", "🎨 Apparence", "📧 Notifications"])
        with tab1:
            self.render_profile_settings()
        with tab2:
            self.render_security_settings()
        with tab3:
            self.render_appearance_settings()
        with tab4:
            self.render_notification_settings()

    def render_profile_settings(self):
        user = st.session_state.user
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("Prénom", value=user['first_name'])
                last_name = st.text_input("Nom", value=user['last_name'])
                email = st.text_input("Email", value=user['email'])
            with col2:
                phone = st.text_input("Téléphone", value=user.get('phone', ''))
                department = st.text_input("Département", value=user.get('department', ''))
                position = st.text_input("Poste", value=user.get('position', ''))
            if st.form_submit_button("💾 Mettre à jour le profil"):
                st.success("Profil mis à jour")

    def render_security_settings(self):
        with st.form("password_form"):
            old_password = st.text_input("Ancien mot de passe", type="password")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            if st.form_submit_button("🔑 Changer le mot de passe"):
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                else:
                    success, msg = self.auth.change_password(
                        st.session_state.user['id'],
                        old_password,
                        new_password
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        st.divider()
        st.subheader("Sessions actives")
        st.info("Aucune autre session active")

    def render_appearance_settings(self):
        theme = st.selectbox("Thème", ["Clair", "Sombre", "Système"])
        language = st.selectbox("Langue", ["Français", "English", "Español"])
        if st.button("💾 Appliquer"):
            st.success("Paramètres sauvegardés")
            st.rerun()

    def render_notification_settings(self):
        email_notif = st.checkbox("Notifications par email", value=True)
        desktop_notif = st.checkbox("Notifications bureau", value=True)
        st.multiselect(
            "Notifications à recevoir",
            ["Maintenances", "Interventions", "Stock", "Rapports"],
            default=["Maintenances", "Interventions", "Stock"]
        )
        if st.button("💾 Sauvegarder"):
            st.success("Préférences sauvegardées")

    def render_footer(self):
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("© 2024 GMAO Enterprise - Version 3.0.0")
        with col2:
            if st.session_state.authenticated:
                st.caption(f"Dernière activité: {st.session_state.last_activity.strftime('%H:%M:%S')}")
        with col3:
            response_time = time.time() - self.start_time
            st.caption(f"Temps de réponse: {response_time:.2f}s")

    def logout(self):
        if st.session_state.session_id:
            self.auth.invalidate_session(st.session_state.session_id)
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_id = None
        st.rerun()

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        app = GMAOApplication()
        app.run()
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}")
        logger.critical(traceback.format_exc())
        st.error("""
        ⚠️ **Erreur critique**

        L'application n'a pas pu démarrer correctement.

        **Causes possibles:**
        - Base de données corrompue ou inaccessible
        - Problème de permissions
        - Dépendances manquantes

        **Solutions:**
        1. Vérifiez que le dossier `data` est accessible en écriture
        2. Supprimez le fichier `data/gmao.db` pour le recréer
        3. Réinstallez les dépendances: `pip install -r requirements.txt`

        **Détails techniques:**
        """)
        st.exception(e)
