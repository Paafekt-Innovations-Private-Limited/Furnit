*"----------------------------------------------------------------------
*"* Wrapper for legacy FM BAPI_SALESORDER_CREATEFROMDAT2
*"* Purpose: create sales order
*"* Modern API: I_SalesOrderTP (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_sample_bapi_salesorder_createfromdat2_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'BAPI_SALESORDER_CREATEFROMDAT2'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_sample_bapi_salesorder_createfromdat2_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
