*"----------------------------------------------------------------------
*"* Wrapper for legacy FM BAPI_ACC_DOCUMENT_POST
*"* Purpose: post FI journal entry
*"* Modern API: I_JournalEntryTP (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_zsample_bapi_acc_document_post_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'BAPI_ACC_DOCUMENT_POST'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_bapi_acc_document_post_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
