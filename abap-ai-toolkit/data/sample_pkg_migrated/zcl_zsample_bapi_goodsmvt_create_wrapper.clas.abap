*"----------------------------------------------------------------------
*"* Wrapper for legacy FM BAPI_GOODSMVT_CREATE
*"* Purpose: create material document
*"* Modern API: I_MaterialDocumentTP (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_zsample_bapi_goodsmvt_create_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'BAPI_GOODSMVT_CREATE'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_bapi_goodsmvt_create_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
