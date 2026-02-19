*"----------------------------------------------------------------------
*"* Wrapper for legacy FM BAPI_PO_CREATE1
*"* Purpose: create purchase order
*"* Modern API: I_PurchaseOrderTP_2 (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_zsample_bapi_po_create1_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'BAPI_PO_CREATE1'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_bapi_po_create1_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
